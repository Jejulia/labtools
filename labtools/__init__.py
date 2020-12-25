"""Various tools for Agilent MS-based quantification"""
from . import model, plot, tidy, pciis, optimizer, example, __main__

"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
"""