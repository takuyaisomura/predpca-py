from .pcacov import pcacov

# Check if torch is installed
try:
    import torch  # noqa: F401

    _has_torch = True
except ImportError:
    _has_torch = False

if _has_torch:
    from .device_manager import DeviceManager  # noqa: F401
    from .pcacov_torch import pcacov as pcacov_torch  # noqa: F401
    from .sym_positive_torch import sym_positive  # noqa: F401

    __all__ = ["DeviceManager", "pcacov", "pcacov_torch", "sym_positive"]
else:
    __all__ = ["pcacov"]
