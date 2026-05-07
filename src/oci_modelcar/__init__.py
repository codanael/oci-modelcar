"""Stream HuggingFace models into OCI registries as multi-layer images."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("oci-modelcar")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
