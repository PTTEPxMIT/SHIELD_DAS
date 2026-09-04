"""Helper called by FINISH_SUPABASE_SETUP.bat.

Reads SB_URL / SB_SERVICE / SB_ANON from the environment (typed into the
.bat's prompts) and writes:

- ``~/.shield_das_publisher.json`` — publisher config (URL + service key
  fallback; the SHIELD_SUPABASE_KEY env var takes precedence when set)
- ``site/config.js`` — viewer-site coordinates (URL + anon key only)
"""

import json
import os
import pathlib
import re
import sys

url = os.environ["SB_URL"].strip().rstrip("/")
service = os.environ["SB_SERVICE"].strip()
anon = os.environ["SB_ANON"].strip()

if not re.fullmatch(r"https://[a-z0-9-]+\.supabase\.co", url):
    sys.exit(f"That does not look like a Supabase project URL: {url!r}")
if not service or not anon:
    sys.exit("Both keys are required.")
if service == anon:
    sys.exit("The service_role and anon keys are identical - check the paste.")

cfg_path = pathlib.Path.home() / ".shield_das_publisher.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
cfg["supabase_url"] = url
cfg["supabase_key"] = service
cfg.setdefault("results_dir", r"C:\Users\remidm\Documents\JD\SHIELD_DAS\results")
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"  wrote {cfg_path}")

site = pathlib.Path(__file__).resolve().parent / "site" / "config.js"
text = site.read_text()
text = re.sub(r'supabaseUrl: "[^"]*"', f'supabaseUrl: "{url}"', text)
text = re.sub(r'supabaseAnonKey: "[^"]*"', f'supabaseAnonKey: "{anon}"', text)
site.write_text(text)
print(f"  wrote {site} (anon key only - safe to publish)")
