import logging

try:
    from .candle import *
except ImportError as e:
    # If we are in development mode, or we did not bundle the CUDA DLLs, we try to locate them here
    logging.warning("CUDA DLLs were not bundled with this package. Trying to locate them...")
    import os
    import platform

    # Try to locate CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH", None)
    if cuda_path:
        logging.warning(f"Found CUDA_PATH environment variable: {cuda_path}")
        if platform.system() == "Windows":
            cuda_path = os.path.join(cuda_path, "bin")
        else:
            cuda_path = os.path.join(cuda_path, "lib64")

        logging.warning(f"Adding {cuda_path} to DLL search path...")
        os.add_dll_directory(cuda_path)

    try:
        from .candle import *
    except ImportError as inner_e:
        raise ImportError("Could not locate CUDA DLLs. Please check the documentation for more information.")

__doc__ = candle.__doc__
if hasattr(candle, "__all__"):
    __all__ = candle.__all__
