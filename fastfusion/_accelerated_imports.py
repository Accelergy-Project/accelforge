import os

os.environ["FASTFUSION_ACCELERATED_IMPORTS"] = "0"

if os.environ.get("FASTFUSION_ACCELERATED_IMPORTS", "0") == "1":
    import cudf as pd
    import cupy as np
    import cupy as scipy

    ACCELERATED = True
else:
    import pandas as pd
    import numpy as np
    import scipy

    ACCELERATED = False
