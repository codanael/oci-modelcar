# oci-modelcar

Stream HuggingFace models directly into OCI registries as multi-layer images,
suitable for KServe with native OCI image volumes (KEP-4639).

## Install

```bash
pip install oci-modelcar
```

## Quick start

```bash
export HF_TOKEN=hf_...
export OCI_USERNAME=...
export OCI_PASSWORD=...

oci-modelcar push \
  --hf-repo Qwen/Qwen3-30B-A3B \
  --registry registry.example.com \
  --target-repo models/qwen3-30b
```

See `oci-modelcar push --help` for all options.

## License

MIT

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml` and update `CHANGELOG.md`.
2. Tag: `git tag v0.1.0 && git push --tags`.
3. The `release.yml` workflow builds, publishes to PyPI via Trusted Publishing,
   and creates a GitHub Release.

PyPI trusted publisher must be configured once: on pypi.org -> Project
Settings -> Publishing -> Add publisher with:
- Owner: `codanael`
- Repo: `oci-modelcar`
- Workflow: `release.yml`
- Environment: `pypi`
