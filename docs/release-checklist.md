# Public Alpha release checklist

Use this gate before changing the GitHub repository from private to public.

- Install from a fresh environment with `python -m pip install .` and run the README example.
- Run the README example on a blank image and verify it handles `best is None` without an
  exception.
- Run `pytest`, `ruff check .`, and all three benchmark programs with their assertion flags.
- Run the optional SmartDoc sample geometry gate and keep its downloaded frames under
  `tmp/`; do not use its quadrilaterals to calibrate chromatic recovery confidence.
- Repeat the 24/48 MP memory smoke on a constrained host whenever artifact mapping changes;
  evidence and overlays must remain at the configured presentation size.
- Verify an oversized burst is rejected before decode, and keep the 2–12 frame, 80 MP
  aggregate and 100 MiB per-file safety limits documented with any config change.
- Confirm every distributed image and third-party asset has recorded redistribution rights.
  Local-only fixtures under `tests/fixtures/local/` must remain untracked.
- Package from tracked files with `powershell -File scripts/package_source.ps1` and inspect the
  resulting archive rather than uploading a working-directory archive.
- Push only history that is allowed to be redistributed. Do not push the private development
  history containing the removed blueprint-derived fixture.
- Before the first public commit, configure the intended GitHub noreply identity and create a
  new root history from the current tracked candidate. Do not push the existing local
  `v0.3.0a3` ref, `--all`, `--tags` or `--mirror`.
- Commit and tag the release candidate before running `scripts/package_source.ps1`; it uses
  `git archive HEAD` and intentionally excludes uncommitted working-tree changes.
- Enable GitHub Private Vulnerability Reporting and Dependabot alerts/security updates, then
  verify the link from `SECURITY.md` while signed out.
- Enable branch protection for `main`, require the lint, test and benchmarks jobs, and
  disallow direct force pushes.
- Confirm Actions default workflow permissions are read-only in repository settings; workflow
  files also declare `contents: read` as defense in depth.
- Verify the license, limitations, version, release notes, clean Git status, and absence of
  secrets one final time.

Camera confidence is not eligible for calibration until the source-group minimum in
[`dataset-protocol.md`](dataset-protocol.md) is met.
