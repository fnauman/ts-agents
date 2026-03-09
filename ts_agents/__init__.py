"""Public package entrypoints for ts-agents."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("ts-agents")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
