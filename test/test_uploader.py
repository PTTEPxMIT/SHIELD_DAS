"""Tests for the shield_das.uploader outbox sweeper.

No network and no hardware: transport tests mock subprocess.run and
urllib.request.urlopen.
"""

import io
import json
import os
import subprocess
import time
from datetime import datetime, timedelta

import pytest

from shield_das import uploader
from shield_das.uploader import (
    UploaderConfig,
    csv_sha256,
    find_completed_runs,
    load_ledger,
    normalize_run,
    push_run,
    run_key,
    sweep,
)

# =============================================================================
# Helpers
# =============================================================================

RUN_START = datetime(2025, 8, 1, 10, 0, 0)


def make_run(
    results_dir,
    date_dir="25.08.01",
    run_name="run_1_10h00",
    date="2025-08-01",
    end_time=None,
    duration_minutes=10.0,
    with_metadata=True,
    corrupt_metadata=False,
    with_backup=False,
    stale=True,
    run_type="permeation_exp",
):
    """Create a fake run directory and return its path.

    Args:
        results_dir: Root results directory.
        date_dir: Name of the date directory (MM.DD or YY.MM.DD form).
        run_name: Name of the run directory.
        date: run_info.date value (%Y-%m-%d).
        end_time: run_info.end_time value, omitted if None.
        duration_minutes: Span between first and last CSV timestamps.
        with_metadata: Whether to write run_metadata.json.
        corrupt_metadata: Write invalid JSON as metadata.
        with_backup: Create a backup/ subdirectory with a file.
        stale: Backdate the CSV mtime by 2 hours.
        run_type: run_info.run_type value.
    """
    run_dir = os.path.join(results_dir, date_dir, run_name)
    os.makedirs(run_dir)

    end = RUN_START + timedelta(minutes=duration_minutes)
    csv_lines = [
        "RealTimestamp,TestGauge_Voltage (V)",
        f"{RUN_START:%Y-%m-%d %H:%M:%S}.000,5.0",
        f"{RUN_START + timedelta(minutes=duration_minutes / 2):%Y-%m-%d %H:%M:%S}"
        ".000,5.1",
        f"{end:%Y-%m-%d %H:%M:%S}.000,5.2",
    ]
    csv_path = os.path.join(run_dir, "shield_data.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    if corrupt_metadata:
        with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
            f.write("not valid json {")
    elif with_metadata:
        run_info = {
            "date": date,
            "start_time": f"{date} 10:00:00",
            "run_type": run_type,
            "data_filename": "shield_data.csv",
        }
        if end_time is not None:
            run_info["end_time"] = end_time
        with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
            json.dump({"version": "1.3", "run_info": run_info}, f)

    if with_backup:
        backup_dir = os.path.join(run_dir, "backup")
        os.makedirs(backup_dir)
        with open(os.path.join(backup_dir, "shield_data_backup_data_1.csv"), "w") as f:
            f.write("backup contents\n")

    if stale:
        old = time.time() - 2 * 3600
        os.utime(csv_path, (old, old))

    return run_dir


@pytest.fixture
def results_dir(tmp_path):
    """Root results directory for fake runs."""
    path = tmp_path / "results"
    path.mkdir()
    return str(path)


@pytest.fixture
def staging_dir(tmp_path):
    """Staging directory for normalised runs."""
    return str(tmp_path / "staging")


@pytest.fixture
def config(results_dir, staging_dir):
    """UploaderConfig pointing at the temp results/staging directories."""
    return UploaderConfig(
        results_dir=results_dir,
        staging_dir=staging_dir,
        min_age_minutes=30.0,
        min_duration_minutes=5.0,
    )


# =============================================================================
# Tests for completion detection
# =============================================================================


def test_find_completed_runs_detects_run_with_end_time(results_dir):
    """A run whose metadata has end_time is complete even with a fresh CSV."""
    run_dir = make_run(results_dir, end_time="2025-08-01 10:10:00", stale=False)
    found = list(find_completed_runs(results_dir))
    assert found == [os.path.abspath(run_dir)]


def test_find_completed_runs_skips_fresh_run_without_end_time(results_dir):
    """A run without end_time whose CSV was just written is not complete."""
    make_run(results_dir, stale=False)
    assert list(find_completed_runs(results_dir)) == []


def test_find_completed_runs_detects_stale_run_without_end_time(results_dir):
    """A run without end_time is complete once its CSV mtime is old enough."""
    run_dir = make_run(results_dir, stale=True)
    found = list(find_completed_runs(results_dir, min_age_minutes=30))
    assert found == [os.path.abspath(run_dir)]


def test_find_completed_runs_skips_too_short_run(results_dir):
    """Runs shorter than min_duration_minutes are skipped."""
    make_run(results_dir, end_time="2025-08-01 10:02:00", duration_minutes=2.0)
    assert list(find_completed_runs(results_dir, min_duration_minutes=5)) == []


def test_find_completed_runs_keeps_short_leak_test(results_dir):
    """A 2-minute leak_test run passes the leak-specific duration floor."""
    run_dir = make_run(
        results_dir,
        end_time="2025-08-01 10:02:00",
        duration_minutes=2.0,
        run_type="leak_test",
    )
    found = list(
        find_completed_runs(
            results_dir, min_duration_minutes=5, min_leak_duration_minutes=1
        )
    )
    assert found == [os.path.abspath(run_dir)]


def test_find_completed_runs_skips_too_short_leak_test(results_dir):
    """A leak test shorter than min_leak_duration_minutes is still skipped."""
    make_run(
        results_dir,
        end_time="2025-08-01 10:00:30",
        duration_minutes=0.5,
        run_type="leak_test",
    )
    found = list(
        find_completed_runs(
            results_dir, min_duration_minutes=5, min_leak_duration_minutes=1
        )
    )
    assert found == []


def test_uploader_config_default_min_leak_duration():
    """UploaderConfig defaults min_leak_duration_minutes to 1.0."""
    assert UploaderConfig().min_leak_duration_minutes == 1.0


def test_find_completed_runs_skips_test_runs(results_dir):
    """test_run_* directories (test mode) are never uploaded."""
    make_run(
        results_dir,
        run_name="test_run_1_10h00",
        end_time="2025-08-01 10:10:00",
    )
    assert list(find_completed_runs(results_dir)) == []


def test_find_completed_runs_skips_missing_metadata(results_dir, caplog):
    """A run without run_metadata.json is skipped with a warning, not raised."""
    make_run(results_dir, with_metadata=False)
    assert list(find_completed_runs(results_dir)) == []
    assert "unreadable metadata" in caplog.text


def test_find_completed_runs_skips_corrupt_metadata(results_dir, caplog):
    """A run with invalid JSON metadata is skipped with a warning, not raised."""
    make_run(results_dir, corrupt_metadata=True)
    assert list(find_completed_runs(results_dir)) == []
    assert "unreadable metadata" in caplog.text


def test_find_completed_runs_walks_both_date_dir_formats(results_dir):
    """Both MM.DD and YY.MM.DD date directories are scanned."""
    old_style = make_run(results_dir, date_dir="08.01", end_time="2025-08-01 10:10:00")
    new_style = make_run(
        results_dir, date_dir="25.08.02", end_time="2025-08-02 10:10:00"
    )
    found = list(find_completed_runs(results_dir))
    assert sorted(found) == sorted(
        [os.path.abspath(old_style), os.path.abspath(new_style)]
    )


def test_find_completed_runs_ignores_unrelated_directories(results_dir):
    """Non-date directories and stray files are ignored."""
    os.makedirs(os.path.join(results_dir, "notes"))
    with open(os.path.join(results_dir, "readme.txt"), "w") as f:
        f.write("hello")
    assert list(find_completed_runs(results_dir)) == []


def test_find_completed_runs_missing_results_dir_does_not_raise(tmp_path):
    """A nonexistent results directory yields nothing rather than raising."""
    assert list(find_completed_runs(str(tmp_path / "nope"))) == []


# =============================================================================
# Tests for run_key and normalization
# =============================================================================


def test_run_key_uses_new_style_date_dir_directly(results_dir):
    """YY.MM.DD date directories map straight into the run key."""
    run_dir = make_run(results_dir, date_dir="25.08.01")
    assert run_key(run_dir) == "25.08.01_run_1_10h00"


def test_run_key_adds_year_from_metadata_for_old_style_dir(results_dir):
    """MM.DD date directories get the year from run_info.date."""
    run_dir = make_run(results_dir, date_dir="08.01", date="2025-08-01")
    assert run_key(run_dir) == "25.08.01_run_1_10h00"


def test_run_key_raises_for_old_style_dir_without_metadata(results_dir):
    """Old-style dirs need metadata for the year; missing metadata raises."""
    run_dir = make_run(results_dir, date_dir="08.01", with_metadata=False)
    with pytest.raises(ValueError, match="Cannot determine year"):
        run_key(run_dir)


def test_normalize_run_creates_staged_directory(results_dir, staging_dir):
    """The staged directory is named YY.MM.DD_run_N_HHhMM."""
    run_dir = make_run(results_dir)
    staged = normalize_run(run_dir, staging_dir)
    assert os.path.basename(staged) == "25.08.01_run_1_10h00"
    assert os.path.isdir(staged)


def test_normalize_run_converts_csv_to_parquet(results_dir, staging_dir):
    """shield_data.csv is converted to measurements.parquet, exactly."""
    pd = pytest.importorskip("pandas")
    run_dir = make_run(results_dir)
    staged = normalize_run(run_dir, staging_dir)
    parquet = os.path.join(staged, "measurements.parquet")
    assert os.path.isfile(parquet)
    assert not os.path.exists(os.path.join(staged, "shield_data.csv"))
    assert not os.path.exists(os.path.join(staged, "pressure_gauge_data.csv"))

    df = pd.read_parquet(parquet)
    source = pd.read_csv(os.path.join(run_dir, "shield_data.csv"))
    assert len(df) == len(source)
    assert df["TestGauge_Voltage (V)"].equals(source["TestGauge_Voltage (V)"])
    assert pd.api.types.is_datetime64_any_dtype(df["RealTimestamp"])


def test_normalize_run_aborts_on_failed_round_trip(
    results_dir, staging_dir, monkeypatch
):
    """A conversion that fails verification raises and removes the parquet."""
    pd = pytest.importorskip("pandas")
    run_dir = make_run(results_dir)
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame())
    with pytest.raises(RuntimeError, match="round-trip failed"):
        normalize_run(run_dir, staging_dir)
    staged = os.path.join(staging_dir, "25.08.01_run_1_10h00")
    assert not os.path.exists(os.path.join(staged, "measurements.parquet"))


def test_normalize_run_leaves_source_csv_untouched(results_dir, staging_dir):
    """The rig's own shield_data.csv is never modified by staging."""
    run_dir = make_run(results_dir)
    csv_path = os.path.join(run_dir, "shield_data.csv")
    with open(csv_path, "rb") as f:
        before = f.read()
    normalize_run(run_dir, staging_dir)
    with open(csv_path, "rb") as f:
        assert f.read() == before


def test_normalize_run_copies_metadata(results_dir, staging_dir):
    """run_metadata.json is copied into the staged directory."""
    run_dir = make_run(results_dir)
    staged = normalize_run(run_dir, staging_dir)
    assert os.path.isfile(os.path.join(staged, "run_metadata.json"))


def test_normalize_run_excludes_backup_directory(results_dir, staging_dir):
    """The backup/ subdirectory is not copied into staging."""
    run_dir = make_run(results_dir, with_backup=True)
    staged = normalize_run(run_dir, staging_dir)
    assert not os.path.exists(os.path.join(staged, "backup"))


def test_normalize_run_handles_old_style_date_dir(results_dir, staging_dir):
    """Old MM.DD runs are staged under the full YY.MM.DD key."""
    run_dir = make_run(results_dir, date_dir="08.01", date="2025-08-01")
    staged = normalize_run(run_dir, staging_dir)
    assert os.path.basename(staged) == "25.08.01_run_1_10h00"


def test_normalize_run_replaces_existing_staged_copy(results_dir, staging_dir):
    """Re-normalising a run replaces any previous staged copy."""
    run_dir = make_run(results_dir)
    staged_first = normalize_run(run_dir, staging_dir)
    marker = os.path.join(staged_first, "stale_file.txt")
    with open(marker, "w") as f:
        f.write("left over")
    staged_second = normalize_run(run_dir, staging_dir)
    assert staged_second == staged_first
    assert not os.path.exists(marker)


# =============================================================================
# Tests for ledger idempotency
# =============================================================================


def _patch_transport(monkeypatch, calls):
    """Replace push_run in the module with a recorder returning a fake URL."""

    def fake_push_run(staged_dir, config):
        calls.append(staged_dir)
        return "https://github.com/PTTEPxMIT/SHIELD-Data/pull/1"

    monkeypatch.setattr(uploader, "push_run", fake_push_run)


def test_sweep_uploads_new_run_and_records_ledger(config, monkeypatch):
    """A new completed run is uploaded and recorded in the ledger."""
    calls = []
    _patch_transport(monkeypatch, calls)
    make_run(config.results_dir, end_time="2025-08-01 10:10:00")

    uploaded = sweep(config)

    assert uploaded == ["25.08.01_run_1_10h00"]
    assert len(calls) == 1
    ledger = load_ledger(config.results_dir)
    entry = ledger["25.08.01_run_1_10h00"]
    assert entry["pr_url"] == "https://github.com/PTTEPxMIT/SHIELD-Data/pull/1"
    assert "sha256" in entry
    assert "uploaded_at" in entry


def test_sweep_is_idempotent_for_unchanged_run(config, monkeypatch):
    """A second sweep with an unchanged CSV uploads nothing."""
    calls = []
    _patch_transport(monkeypatch, calls)
    make_run(config.results_dir, end_time="2025-08-01 10:10:00")

    assert sweep(config) == ["25.08.01_run_1_10h00"]
    assert sweep(config) == []
    assert len(calls) == 1


def test_sweep_reuploads_when_csv_changes(config, monkeypatch):
    """A changed CSV hash triggers a re-upload (superseding PR)."""
    calls = []
    _patch_transport(monkeypatch, calls)
    run_dir = make_run(config.results_dir, end_time="2025-08-01 10:10:00")
    sweep(config)

    csv_path = os.path.join(run_dir, "shield_data.csv")
    with open(csv_path, "a") as f:
        f.write("2025-08-01 10:11:00.000,5.3\n")
    old = time.time() - 2 * 3600
    os.utime(csv_path, (old, old))

    assert sweep(config) == ["25.08.01_run_1_10h00"]
    assert len(calls) == 2


def test_sweep_continues_after_failed_upload(config, monkeypatch, caplog):
    """A transport failure for one run is logged and does not abort the sweep."""

    def failing_push_run(staged_dir, config):
        raise RuntimeError("push failed")

    monkeypatch.setattr(uploader, "push_run", failing_push_run)
    make_run(config.results_dir, end_time="2025-08-01 10:10:00")

    assert sweep(config) == []
    assert "Upload failed" in caplog.text
    assert load_ledger(config.results_dir) == {}


def test_csv_sha256_changes_with_content(config):
    """The ledger hash tracks CSV content."""
    run_dir = make_run(config.results_dir, end_time="2025-08-01 10:10:00")
    before = csv_sha256(run_dir)
    with open(os.path.join(run_dir, "shield_data.csv"), "a") as f:
        f.write("2025-08-01 10:11:00.000,5.3\n")
    assert csv_sha256(run_dir) != before


# =============================================================================
# Tests for dry-run and CLI
# =============================================================================


def test_sweep_dry_run_prints_without_uploading(config, monkeypatch, capsys):
    """Dry-run reports the run but performs no staging, transport, or ledger."""

    def forbidden(*args, **kwargs):
        raise AssertionError("transport must not run during dry-run")

    monkeypatch.setattr(uploader, "push_run", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    make_run(config.results_dir, end_time="2025-08-01 10:10:00")

    uploaded = sweep(config, dry_run=True)

    assert uploaded == ["25.08.01_run_1_10h00"]
    out = capsys.readouterr().out
    assert "would upload: 25.08.01_run_1_10h00" in out
    assert not os.path.exists(config.staging_dir)
    assert load_ledger(config.results_dir) == {}


def test_main_dry_run_with_config_file(tmp_path, results_dir, monkeypatch, capsys):
    """The CLI loads the config file and honours --dry-run."""
    make_run(results_dir, end_time="2025-08-01 10:10:00")
    config_path = tmp_path / "uploader_config.json"
    config_path.write_text(
        json.dumps({"results_dir": results_dir, "min_duration_minutes": 5})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no network during dry-run")

    monkeypatch.setattr(subprocess, "run", forbidden)

    exit_code = uploader.main(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    assert "25.08.01_run_1_10h00" in capsys.readouterr().out


def test_main_reports_nothing_to_upload(tmp_path, results_dir, capsys):
    """The CLI reports when no completed runs are found."""
    config_path = tmp_path / "uploader_config.json"
    config_path.write_text(json.dumps({"results_dir": results_dir}))

    exit_code = uploader.main(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    assert "Nothing to upload" in capsys.readouterr().out


def test_uploader_config_defaults_staging_dir():
    """staging_dir defaults to a hidden directory inside results_dir."""
    cfg = UploaderConfig(results_dir="some_results")
    assert cfg.staging_dir == os.path.join("some_results", ".upload_staging")


def test_uploader_config_from_missing_file_uses_defaults(tmp_path):
    """Loading from a nonexistent path returns default configuration."""
    cfg = UploaderConfig.from_file(str(tmp_path / "absent.json"))
    assert cfg.repo == "PTTEPxMIT/SHIELD-Data"
    assert cfg.min_age_minutes == 30.0


def test_uploader_config_from_file_ignores_unknown_keys(tmp_path, caplog):
    """Unknown config keys are ignored with a warning."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"repo": "other/repo", "bogus_key": 1}))
    cfg = UploaderConfig.from_file(str(path))
    assert cfg.repo == "other/repo"
    assert "bogus_key" in caplog.text


# =============================================================================
# Tests for transport (mocked subprocess + urllib)
# =============================================================================

TOKEN = "ghp_SECRETTOKEN12345"


@pytest.fixture
def staged_run(results_dir, staging_dir):
    """A normalised staged run ready for transport."""
    run_dir = make_run(results_dir, end_time="2025-08-01 10:10:00")
    return normalize_run(run_dir, staging_dir)


def _fake_urlopen_factory(responses):
    """Build a fake urlopen returning queued JSON payloads."""

    def fake_urlopen(request):
        responses.setdefault("requests", []).append(request)
        payload = responses["queue"].pop(0)
        return io.BytesIO(json.dumps(payload).encode())

    return fake_urlopen


def test_push_run_clones_branches_commits_and_pushes(config, staged_run, monkeypatch):
    """push_run drives git through clone, branch, add, commit, and push."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)
    git_calls = []

    def fake_run(cmd, **kwargs):
        git_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    responses = {"queue": [{"html_url": "https://example.test/pull/7"}]}
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader.urllib.request, "urlopen", _fake_urlopen_factory(responses)
    )

    pr_url = push_run(staged_run, config)

    assert pr_url == "https://example.test/pull/7"
    subcommands = [cmd[1] if cmd[1] != "-C" else cmd[3] for cmd in git_calls]
    assert subcommands[0] == "clone"
    assert "checkout" in subcommands
    assert "add" in subcommands
    assert "push" in subcommands
    commit_cmd = next(cmd for cmd in git_calls if "commit" in cmd)
    assert "Add run 25.08.01_run_1_10h00 (auto-upload)" in commit_cmd
    clone_cmd = git_calls[0]
    assert any(TOKEN in part for part in clone_cmd)  # token used for auth
    branch_cmd = next(cmd for cmd in git_calls if "checkout" in cmd)
    assert "auto/run-25.08.01_run_1_10h00" in branch_cmd


def test_push_run_opens_pr_with_expected_payload(config, staged_run, monkeypatch):
    """The PR request targets the repo with the right title, head, and base."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    responses = {"queue": [{"html_url": "https://example.test/pull/7"}]}
    monkeypatch.setattr(
        uploader.urllib.request, "urlopen", _fake_urlopen_factory(responses)
    )

    push_run(staged_run, config)

    request = responses["requests"][0]
    assert "api.github.com/repos/PTTEPxMIT/SHIELD-Data/pulls" in request.full_url
    payload = json.loads(request.data.decode())
    assert payload["title"] == "Add run 25.08.01_run_1_10h00"
    assert payload["head"] == "auto/run-25.08.01_run_1_10h00"
    assert payload["base"] == "main"
    assert "auto-uploaded" in payload["body"].lower()


def test_push_run_copies_run_into_run_data(config, staged_run, monkeypatch):
    """The staged run is committed under run_data/<run_key>/ in the clone."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)
    seen = {}

    def fake_run(cmd, **kwargs):
        if "add" in cmd:
            repo_dir = cmd[cmd.index("-C") + 1]
            staged_copy = os.path.join(repo_dir, "run_data", "25.08.01_run_1_10h00")
            seen["parquet"] = os.path.isfile(
                os.path.join(staged_copy, "measurements.parquet")
            )
            seen["metadata"] = os.path.isfile(
                os.path.join(staged_copy, "run_metadata.json")
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    responses = {"queue": [{"html_url": "https://example.test/pull/7"}]}
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader.urllib.request, "urlopen", _fake_urlopen_factory(responses)
    )

    push_run(staged_run, config)

    assert seen == {"parquet": True, "metadata": True}


def test_push_run_never_exposes_token_on_git_failure(
    config, staged_run, monkeypatch, caplog
):
    """A failing git command surfaces redacted output, never the token."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)

    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            128,
            stdout="",
            stderr=f"fatal: could not read from "
            f"https://x-access-token:{TOKEN}@github.com/x/y.git",
        )

    monkeypatch.setattr(subprocess, "run", failing_run)

    with pytest.raises(RuntimeError) as excinfo:
        push_run(staged_run, config)

    assert TOKEN not in str(excinfo.value)
    assert "***" in str(excinfo.value)
    assert TOKEN not in caplog.text


def test_push_run_never_exposes_token_on_api_failure(
    config, staged_run, monkeypatch, caplog
):
    """A GitHub API error surfaces with the token redacted."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )

    def failing_urlopen(request):
        raise uploader.urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(f"bad credentials: {TOKEN}".encode()),
        )

    monkeypatch.setattr(uploader.urllib.request, "urlopen", failing_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        push_run(staged_run, config)

    assert TOKEN not in str(excinfo.value)
    assert TOKEN not in caplog.text


def test_push_run_requires_token(config, staged_run, monkeypatch):
    """push_run raises a clear error when no token is configured."""
    monkeypatch.delenv("SHIELD_UPLOAD_TOKEN", raising=False)
    config.token = None

    with pytest.raises(RuntimeError, match="No upload token"):
        push_run(staged_run, config)


def test_push_run_reuses_existing_pr_on_422(config, staged_run, monkeypatch):
    """If the PR already exists (422), the existing PR URL is returned."""
    monkeypatch.setenv("SHIELD_UPLOAD_TOKEN", TOKEN)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    calls = {"n": 0}

    def urlopen_422_then_list(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise uploader.urllib.error.HTTPError(
                request.full_url,
                422,
                "Unprocessable",
                {},
                io.BytesIO(b'{"message": "A pull request already exists"}'),
            )
        return io.BytesIO(
            json.dumps([{"html_url": "https://example.test/pull/3"}]).encode()
        )

    monkeypatch.setattr(uploader.urllib.request, "urlopen", urlopen_422_then_list)

    assert push_run(staged_run, config) == "https://example.test/pull/3"
