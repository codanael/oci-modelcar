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
