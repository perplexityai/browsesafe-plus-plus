# Contributing

Thanks for helping improve BrowseSafe++.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                     # the suite runs offline in a few seconds
```

## Ground rules

- **Tests run without network access.** Use `browsesafe_pp.testing.write_fixture` for data and `httpx.MockTransport` for model endpoints.
- **Keep the core dependency-light.** Required dependencies are `pyarrow`, `huggingface-hub` and `filelock`; anything else goes behind an extra and is imported lazily.
- **Never hide failures.** Errors, abstentions, invalid outputs and excluded pairs must remain visible in reports and manifests.
- **Keep secrets out of artifacts.** Reports, manifests and configurations must not contain API keys or page text beyond what the user explicitly saves.
- **Keep the fixture harmless.** Test pages are fictional and their injections are inert marker strings.
- **Document public changes** in `docs/` and `CHANGELOG.md`.

## Proposing changes

Open an issue that describes the problem or the change you have in mind, with a minimal example where possible. Changes must keep `pytest` passing on Python 3.10 and newer and come with tests and documentation for anything user-visible.

## Reporting issues with the data

Please include the dataset revision (`benchmark.manifest["revision"]`), the `entry_id` and the view. Do not paste large amounts of page content into public issues.
