# Project Rules

Purpose: Shared guidelines for coding, UI, and workflow in this repository so contributors stay consistent and productive.

## Scope
- Applies to all Python, UI (PyQt6), and documentation changes.
- Keep rules practical. Propose changes via PR if something is blocking.

## Repository Conventions
- Encoding: UTF-8. Newlines: LF.
- Filenames: snake_case for modules, `PascalCase` for classes.
- Large binaries: avoid committing unless essential for demo/testing.
- Secrets: never commit credentials or personal data.

## Python Style
- Version: Python 3.10+ (match local environment).
- Style: PEP 8, 4 spaces, aim for ≤ 100 characters per line.
- Type hints where useful. Public functions should have docstrings.
- Imports: standard lib, third-party, local (grouped and alphabetized).
- Logging: prefer `logging` over `print` in long‑running workflows; `print` is acceptable for temporary debug.

## PyQt6 / UI Rules
- Threading: do not touch widgets from worker threads. Communicate via `pyqtSignal`/`pyqtSlot` or `QTimer` updates.
- Responsiveness: avoid blocking the UI thread; heavy work goes to `QThread`/workers.
- Naming: suffix widgets (e.g., `*_label`, `*_button`, `*_slider`).
- Visual guides: default crosshair color is green. Keep overlays subtle and non‑obstructive.
- Styles: centralize colors and fonts in a single `set_stylesheet()` helper.
- Accessibility: maintain readable contrast; avoid tiny text (< 12pt).

## Error Handling
- Validate camera/video sources; show clear messages instead of crashing.
- Wrap I/O and OpenCV calls; fail gracefully and log the error context.

## Tests & Verification
- Add focused tests for pure functions when feasible.
- For UI changes, include before/after screenshots or a short clip in PRs.

## Git Workflow
- Branch names: `feature/<short-name>`, `fix/<issue-id>`, `chore/<scope>`.
- Commit messages (present tense):
  - Good: `feat(ui): add color picker for crosshair`
  - Good: `fix(video): handle missing codec gracefully`
- One logical change per commit when possible.

## Pull Requests
- Description: what/why/how, risks, screenshots for UI.
- Keep PRs small and focused; link related issues.
- Checklist:
  - Code builds and runs locally.
  - No UI freezes; long tasks moved off main thread.
  - Names, comments, and docs are clear.

## Versioning & Releases
- Use semantic versioning when tagging releases: MAJOR.MINOR.PATCH.
- Document notable changes in `CHANGELOG.md` (if introduced).

## Documentation
- Keep `manual.txt` user‑focused; keep `rule.md` contributor‑focused.
- Update inline comments and docstrings when behavior changes.

## Proposing Rule Changes
- Open a PR titled `docs(rule): <summary>` describing the change and rationale.
- Keep the rule set short and actionable.

— End —

