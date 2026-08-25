# Deterministic fixtures

These files are abstract, locally generated grayscale PNG patterns. They are
not clinical radiographs, do not contain PHI, and must not be used for
diagnosis or model-quality claims.

Regenerate the bytes and manifest with:

```bash
uv run --group dev python scripts/generate_fixtures.py
```

`manifest.json` is the source of truth for each fixture's SHA-256 digest, media
type, byte size, dimensions, synthetic origin, license status, and intended test
use. Tests verify the manifest against the files on disk.

Fixtures:

- `synthetic-full-frame.png`: full-frame locator and response tests.
- `synthetic-crop-target.png`: full-frame and normalized crop locator tests.
- `synthetic-abstention.png`: insufficient-evidence and abstention tests.

Do not replace these with downloaded medical images. Any future image must be
synthetic or have explicit redistribution clearance recorded in the manifest.
