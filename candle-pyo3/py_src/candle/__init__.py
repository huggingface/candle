import logging

try:
    from .candle import *
except ImportError as e:
    # If we are in development mode, or we did not bundle the DLLs, we try to locate them here
    # PyO3 wont give us any information about what DLLs are missing, so we can only try to load
    # the DLLs and re-import the module
    logging.warning("DLLs were not bundled with this package. Trying to locate them...")
    import os
    import platform

    def locate_cuda_dlls():
        logging.warning("Locating CUDA DLLs...")
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
        else:
            logging.warning("CUDA_PATH environment variable not found!")

    def locate_mkl_dlls():
        # Try to locate ONEAPI_ROOT environment variable
        oneapi_root = os.environ.get("ONEAPI_ROOT", None)
        if oneapi_root:
            if platform.system() == "Windows":
                mkl_path = os.path.join(
                    oneapi_root, "compiler", "latest", "windows", "redist", "intel64_win", "compiler"
                )
            else:
                mkl_path = os.path.join(oneapi_root, "mkl", "latest", "lib", "intel64")

            logging.warning(f"Adding {mkl_path} to DLL search path...")
            os.add_dll_directory(mkl_path)
        else:
            logging.warning("ONEAPI_ROOT environment variable not found!")

    locate_cuda_dlls()
    locate_mkl_dlls()

    try:
        from .candle import *
    except ImportError as inner_e:
        raise ImportError("Could not locate DLLs. Please check the documentation for more information.")

__doc__ = candle.__doc__
if hasattr(candle, "__all__"):
    __all__ = candle.__all__
