from __future__ import annotations

import os
import warnings


def configure_runtime_environment() -> None:
    if os.name == "nt":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3.*charset_normalizer.*doesn't match a supported version.*",
        category=Warning,
    )
