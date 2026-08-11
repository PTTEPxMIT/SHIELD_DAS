"""Outbox sweeper that uploads completed runs to the SHIELD-Data repository.

Scans the local ``results/`` directory for completed (non test-mode) runs,
normalises each run into the SHIELD-Data layout
(``run_data/YY.MM.DD_run_N_HHhMM/`` with the CSV renamed to
``pressure_gauge_data.csv``), and opens a pull request against
`PTTEPxMIT/SHIELD-Data <https://github.com/PTTEPxMIT/SHIELD-Data>`_ for each
new or changed run. A ledger file (``<results_dir>/.upload_ledger.json``)
keyed by the sha256 of the run's CSV makes the sweep idempotent, so it is safe
to run any number of times.

Run by hand after a run finishes — ``upload_runs.bat`` in the repo root
double-click-runs it on the rig PC. Uses only the Python standard library plus
the ``git`` CLI. See ``docs/auto_upload.md`` for setup instructions (token and
config file).
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, fields
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".shield_das_uploader.json")
LEDGER_FILENAME = ".upload_ledger.json"
DATA_CSV_NAME = "shield_data.csv"
UPLOAD_CSV_NAME = "pressure_gauge_data.csv"
METADATA_NAME = "run_metadata.json"
TOKEN_ENV_VAR = "SHIELD_UPLOAD_TOKEN"

# Date directories: old rigs use MM.DD, newer ones YY.MM.DD
_DATE_DIR_OLD_RE = re.compile(r"^\d{2}\.\d{2}$")
_DATE_DIR_NEW_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
# Run directories: run_<number>_<HHhMM>. Test runs are prefixed test_run_ and
# deliberately do not match.
_RUN_DIR_RE = re.compile(r"^run_(\d+)_(\d{1,2}h\d{2})$")


@dataclass
class UploaderConfig:
    """Configuration for the run uploader.

    Attributes:
        results_dir: Local directory containing recorded runs
            (``<date>/<run>/`` subdirectories).
        repo: GitHub repository receiving the runs, as ``owner/name``.
        staging_dir: Directory where normalised copies of runs are staged
            before upload. Defaults to ``<results_dir>/.upload_staging``.
        min_age_minutes: A run without an ``end_time`` in its metadata is
            considered complete once its CSV has not been modified for this
            many minutes.
        min_duration_minutes: Runs shorter than this (first-to-last CSV
            timestamp, in minutes) are skipped as aborted starts.
        token: GitHub token used for pushing and opening pull requests. The
            ``SHIELD_UPLOAD_TOKEN`` environment variable takes precedence.
    """

    results_dir: str = "results"
    repo: str = "PTTEPxMIT/SHIELD-Data"
    staging_dir: str | None = None
    min_age_minutes: float = 30.0
    min_duration_minutes: float = 5.0
    token: str | None = None

    def __post_init__(self):
        if self.staging_dir is None:
            self.staging_dir = os.path.join(self.results_dir, ".upload_staging")

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> "UploaderConfig":
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON config file. If the file does not exist,
                defaults are used.

        Returns:
            The loaded UploaderConfig.
        """
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown config keys in %s: %s", path, unknown)

        return cls(**{k: v for k, v in data.items() if k in known})


def _read_metadata(run_dir: str) -> dict | None:
    """Read run_metadata.json from a run directory.

    Args:
        run_dir: Path to the run directory.

    Returns:
        The parsed metadata dict, or None (with a warning logged) if the file
        is missing or corrupt.
    """
    metadata_path = os.path.join(run_dir, METADATA_NAME)
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping %s: unreadable metadata (%s)", run_dir, exc)
        return None


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a CSV RealTimestamp value.

    Args:
        value: Timestamp string, with or without milliseconds.

    Returns:
        The parsed datetime, or None if the value is not a timestamp.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _run_duration_minutes(csv_path: str) -> float | None:
    """Compute run duration from the first and last CSV data rows.

    Only the head and tail of the file are read; the full CSV is never
    loaded.

    Args:
        csv_path: Path to the run's data CSV.

    Returns:
        Duration in minutes, or None if it cannot be determined.
    """
    try:
        with open(csv_path, "rb") as f:
            f.readline()  # header
            first_line = f.readline().decode("utf-8", errors="replace")
            if not first_line.strip():
                return None

            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail_lines = [
                line
                for line in f.read().decode("utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if not tail_lines:
                return None
            last_line = tail_lines[-1]
    except OSError as exc:
        logger.warning("Could not read %s: %s", csv_path, exc)
        return None

    first_ts = _parse_timestamp(first_line.split(",", 1)[0].strip())
    last_ts = _parse_timestamp(last_line.split(",", 1)[0].strip())
    if first_ts is None or last_ts is None:
        return None
    return (last_ts - first_ts).total_seconds() / 60.0


def find_completed_runs(
    results_dir: str,
    min_age_minutes: float = 30.0,
    min_duration_minutes: float = 5.0,
) -> Iterator[str]:
    """Yield run directories that are complete and worth uploading.

    Walks date directories in both the old ``MM.DD`` and new ``YY.MM.DD``
    naming forms. A run qualifies when all of the following hold:

    - it is not a test-mode run (``test_run_*`` directories are skipped);
    - its metadata has ``run_info.end_time``, OR its ``shield_data.csv`` was
      last modified more than ``min_age_minutes`` ago;
    - its duration (first-to-last CSV timestamp) is at least
      ``min_duration_minutes``.

    Runs with missing or corrupt metadata are skipped with a warning; this
    function never raises for a malformed run.

    Args:
        results_dir: Directory containing date-named run subdirectories.
        min_age_minutes: Age threshold (minutes since last CSV write) used
            when no end_time is recorded.
        min_duration_minutes: Minimum run duration in minutes.

    Yields:
        Absolute paths of completed run directories.
    """
    if not os.path.isdir(results_dir):
        logger.warning("Results directory does not exist: %s", results_dir)
        return

    for date_name in sorted(os.listdir(results_dir)):
        date_dir = os.path.join(results_dir, date_name)
        if not os.path.isdir(date_dir):
            continue
        if not (_DATE_DIR_OLD_RE.match(date_name) or _DATE_DIR_NEW_RE.match(date_name)):
            continue

        for run_name in sorted(os.listdir(date_dir)):
            run_dir = os.path.join(date_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            # test_run_* directories (test mode) never match this pattern
            if not _RUN_DIR_RE.match(run_name):
                continue

            csv_path = os.path.join(run_dir, DATA_CSV_NAME)
            if not os.path.isfile(csv_path):
                logger.warning("Skipping %s: no %s", run_dir, DATA_CSV_NAME)
                continue

            metadata = _read_metadata(run_dir)
            if metadata is None:
                continue

            run_info = metadata.get("run_info", {})
            has_end_time = bool(run_info.get("end_time"))
            age_minutes = (
                datetime.now().timestamp() - os.path.getmtime(csv_path)
            ) / 60.0
            if not has_end_time and age_minutes < min_age_minutes:
                logger.info(
                    "Skipping %s: no end_time and CSV modified %.1f min ago",
                    run_dir,
                    age_minutes,
                )
                continue

            duration = _run_duration_minutes(csv_path)
            if duration is None:
                logger.warning("Skipping %s: could not determine duration", run_dir)
                continue
            if duration < min_duration_minutes:
                logger.info(
                    "Skipping %s: duration %.1f min < %.1f min",
                    run_dir,
                    duration,
                    min_duration_minutes,
                )
                continue

            yield os.path.abspath(run_dir)


def run_key(run_dir: str) -> str:
    """Build the canonical SHIELD-Data key for a run directory.

    The key has the form ``YY.MM.DD_run_N_HHhMM``. For old-style ``MM.DD``
    date directories, the year is taken from ``run_info.date`` in the
    metadata.

    Args:
        run_dir: Path to the run directory.

    Returns:
        The run key string.

    Raises:
        ValueError: If the directory name or metadata cannot be interpreted.
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    if not _RUN_DIR_RE.match(run_name):
        raise ValueError(f"Not a run directory name: {run_name!r}")

    date_name = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    if _DATE_DIR_NEW_RE.match(date_name):
        date_part = date_name
    elif _DATE_DIR_OLD_RE.match(date_name):
        metadata = _read_metadata(run_dir)
        if metadata is None:
            raise ValueError(f"Cannot determine year for {run_dir}: no metadata")
        date_str = metadata.get("run_info", {}).get("date", "")
        try:
            year_two_digit = datetime.strptime(date_str, "%Y-%m-%d").strftime("%y")
        except ValueError as exc:
            raise ValueError(
                f"Cannot determine year for {run_dir}: bad run_info.date {date_str!r}"
            ) from exc
        date_part = f"{year_two_digit}.{date_name}"
    else:
        raise ValueError(f"Unrecognised date directory name: {date_name!r}")

    return f"{date_part}_{run_name}"


def normalize_run(run_dir: str, staging_dir: str) -> str:
    """Copy a run into the staging directory in the SHIELD-Data layout.

    Creates ``<staging_dir>/YY.MM.DD_run_N_HHhMM/`` containing the run's
    files with ``shield_data.csv`` renamed to ``pressure_gauge_data.csv``.
    The ``backup/`` subdirectory is excluded. Any existing staged copy of the
    same run is replaced.

    Args:
        run_dir: Path to the source run directory.
        staging_dir: Directory in which to create the staged copy.

    Returns:
        Path to the staged run directory.
    """
    key = run_key(run_dir)
    staged = os.path.join(staging_dir, key)
    if os.path.exists(staged):
        shutil.rmtree(staged)
    os.makedirs(staging_dir, exist_ok=True)

    shutil.copytree(run_dir, staged, ignore=shutil.ignore_patterns("backup"))

    csv_src = os.path.join(staged, DATA_CSV_NAME)
    os.replace(csv_src, os.path.join(staged, UPLOAD_CSV_NAME))
    return staged


def csv_sha256(run_dir: str) -> str:
    """Compute the sha256 of a run's shield_data.csv content.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Hex digest of the CSV content.
    """
    digest = hashlib.sha256()
    with open(os.path.join(run_dir, DATA_CSV_NAME), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ledger(results_dir: str) -> dict:
    """Load the upload ledger from the results directory.

    Args:
        results_dir: Directory containing the ``.upload_ledger.json`` file.

    Returns:
        The ledger mapping ``run_key -> {"sha256", "uploaded_at", "pr_url"}``.
        Empty if the ledger is missing or corrupt.
    """
    path = os.path.join(results_dir, LEDGER_FILENAME)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read ledger %s (%s); starting fresh", path, exc)
        return {}


def save_ledger(results_dir: str, ledger: dict) -> None:
    """Write the upload ledger to the results directory.

    Args:
        results_dir: Directory in which to write ``.upload_ledger.json``.
        ledger: Ledger mapping to serialise.
    """
    path = os.path.join(results_dir, LEDGER_FILENAME)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)


def _get_token(config: UploaderConfig) -> str:
    """Resolve the GitHub token from the environment or config.

    Args:
        config: Uploader configuration.

    Returns:
        The token string.

    Raises:
        RuntimeError: If no token is configured.
    """
    token = os.environ.get(TOKEN_ENV_VAR) or config.token
    if not token:
        raise RuntimeError(
            f"No upload token: set the {TOKEN_ENV_VAR} environment variable or "
            'the "token" key in the uploader config file'
        )
    return token


def _redact(text: str, token: str) -> str:
    """Remove the token from a string before it is logged or raised.

    Args:
        text: Text that may contain the token.
        token: The secret to strip.

    Returns:
        The text with every occurrence of the token replaced by ``***``.
    """
    if not token:
        return text
    return text.replace(token, "***")


def _git(args: list[str], token: str) -> None:
    """Run a git command, raising with token-redacted output on failure.

    Args:
        args: Arguments passed to git (without the leading ``git``).
        token: Token to redact from any surfaced output.

    Raises:
        RuntimeError: If the command exits non-zero.
    """
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        command = _redact(" ".join(["git", *args]), token)
        output = _redact(f"{result.stdout}\n{result.stderr}".strip(), token)
        raise RuntimeError(f"Command failed ({command}): {output}")


def _github_api(url: str, token: str, payload: dict | None = None) -> dict | list:
    """Call the GitHub REST API.

    Args:
        url: Full API URL.
        token: GitHub token for the Authorization header.
        payload: JSON body; POSTs if given, otherwise GETs.

    Returns:
        The decoded JSON response.

    Raises:
        RuntimeError: On an HTTP error, with the token redacted.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = _redact(exc.read().decode("utf-8", errors="replace"), token)
        raise RuntimeError(f"GitHub API error {exc.code} for {url}: {body}") from None


def push_run(staged_dir: str, config: UploaderConfig) -> str:
    """Push a staged run to SHIELD-Data and open a pull request.

    Shallow-clones the target repository into a temporary directory, creates
    (or force-updates) the branch ``auto/run-<run_key>`` containing the run
    under ``run_data/``, pushes it, and opens a PR via the GitHub REST API.
    If an open PR for the branch already exists, its URL is returned instead.

    The token is never printed; it is stripped from any error output.

    Args:
        staged_dir: Path to a staged run directory (from normalize_run).
        config: Uploader configuration.

    Returns:
        The URL of the pull request.
    """
    key = os.path.basename(os.path.normpath(staged_dir))
    token = _get_token(config)
    branch = f"auto/run-{key}"
    tmp = tempfile.mkdtemp(prefix="shield_upload_")
    try:
        repo_dir = os.path.join(tmp, "SHIELD-Data")
        clone_url = f"https://x-access-token:{token}@github.com/{config.repo}.git"
        _git(["clone", "--depth", "1", clone_url, repo_dir], token)
        _git(["-C", repo_dir, "checkout", "-b", branch], token)

        dest = os.path.join(repo_dir, "run_data", key)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(staged_dir, dest)

        _git(["-C", repo_dir, "add", "run_data"], token)
        _git(
            [
                "-C",
                repo_dir,
                "-c",
                "user.name=SHIELD DAS uploader",
                "-c",
                "user.email=shield-das-uploader@users.noreply.github.com",
                "commit",
                "-m",
                f"Add run {key} (auto-upload)",
            ],
            token,
        )
        # Force so a re-upload of a changed run supersedes the old branch
        _git(["-C", repo_dir, "push", "--force", "-u", "origin", branch], token)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    api_base = f"https://api.github.com/repos/{config.repo}"
    try:
        response = _github_api(
            f"{api_base}/pulls",
            token,
            payload={
                "title": f"Add run {key}",
                "head": branch,
                "base": "main",
                "body": (
                    f"Auto-uploaded from the SHIELD rig by `shield-das-upload`.\n\n"
                    f"Run `{key}` recorded by SHIELD_DAS; CSV renamed to "
                    f"`{UPLOAD_CSV_NAME}`, `backup/` excluded."
                ),
            },
        )
        return response.get("html_url", "")
    except RuntimeError as exc:
        # 422 usually means a PR for this branch already exists; reuse it
        if "422" not in str(exc):
            raise
        owner = config.repo.split("/")[0]
        existing = _github_api(
            f"{api_base}/pulls?head={owner}:{branch}&state=open", token
        )
        if existing:
            url = existing[0].get("html_url", "")
            logger.info("PR already open for %s: %s", key, url)
            return url
        raise


def sweep(config: UploaderConfig, dry_run: bool = False) -> list[str]:
    """Find completed runs and upload any that are new or changed.

    Args:
        config: Uploader configuration.
        dry_run: If True, print what would be uploaded and do nothing else
            (no staging, no network).

    Returns:
        List of run keys that were uploaded (or would be, for a dry run).
    """
    ledger = load_ledger(config.results_dir)
    processed = []

    for run_dir in find_completed_runs(
        config.results_dir,
        min_age_minutes=config.min_age_minutes,
        min_duration_minutes=config.min_duration_minutes,
    ):
        try:
            key = run_key(run_dir)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", run_dir, exc)
            continue

        sha = csv_sha256(run_dir)
        entry = ledger.get(key)
        if entry and entry.get("sha256") == sha:
            logger.info("Already uploaded, unchanged: %s", key)
            continue

        if dry_run:
            action = "re-upload (changed)" if entry else "upload"
            print(f"[dry-run] would {action}: {key}  ({run_dir})")
            processed.append(key)
            continue

        try:
            staged = normalize_run(run_dir, config.staging_dir)
            pr_url = push_run(staged, config)
        except (OSError, RuntimeError) as exc:
            logger.error("Upload failed for %s: %s", key, exc)
            continue

        ledger[key] = {
            "sha256": sha,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "pr_url": pr_url,
        }
        save_ledger(config.results_dir, ledger)
        logger.info("Uploaded %s -> %s", key, pr_url)
        processed.append(key)

    return processed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the run uploader.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        prog="shield-das-upload",
        description=(
            "Upload completed SHIELD runs to the SHIELD-Data repository as "
            "pull requests."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without touching the network",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = UploaderConfig.from_file(args.config)
    processed = sweep(config, dry_run=args.dry_run)
    if not processed:
        print("Nothing to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
