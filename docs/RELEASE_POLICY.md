# Release and support policy

## Cadence
- Minor releases: monthly train.
- Patch releases: on-demand for reliability/security fixes.

## Support window
- Latest minor + previous minor are supported.
- Security fixes are backported to all supported minors.

## Security advisories
- Use private triage first, then coordinated disclosure with fixed version.
- Publish CVE/advisory notes in release notes.

## Verification
```bash
python scripts/release_readiness_report.py --output quality/release_readiness_report.md
```
