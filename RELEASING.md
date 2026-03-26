# Releasing

Zipcoil releases are published to PyPI by GitHub Actions when a GitHub Release is published.

## Prerequisites

- You have push access to `vladiliescu/zipcoil`.
- `PYPI_API_TOKEN` is configured in the GitHub repository secrets.
- Your branch is up to date and CI is green.

## Steps

1. Update the version in `pyproject.toml`.
2. Commit the version bump and push it to `main`.
3. Wait for the test workflow to pass.
4. Create and push a tag using the `vX.Y.Z` format.
5. Publish a GitHub Release for that tag.
6. Confirm that the `Publish to PyPI` workflow succeeds.

## Example

```bash
git add pyproject.toml
git commit -m "build: bump version to 0.3.0"
git push origin main

git tag v0.3.0
git push origin v0.3.0
```

Then publish the `v0.3.0` GitHub Release.

## Notes

- The package version comes from `pyproject.toml`. Do not tag or publish without bumping it first.
- Publishing the release triggers the PyPI upload workflow.
- The workflow can also be started manually with `workflow_dispatch`, but it will publish whatever version exists in the selected ref.
