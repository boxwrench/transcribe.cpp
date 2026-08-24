# Runs

Each script creates a new timestamped directory and refuses to overwrite an
existing one. `run.json` records the command and context; `MANIFEST.sha256`
hashes every payload in the run. These generated directories are ignored.
