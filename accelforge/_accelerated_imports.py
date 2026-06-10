import os

os.environ["ACCELFORGE_ACCELERATED_IMPORTS"] = "0"

if os.environ.get("ACCELFORGE_ACCELERATED_IMPORTS", "0") == "1":
    import cudf as pd
    import cupy as np

    pandas = pd
    numpy = np
    # import cupy as scipy

    ACCELERATED = True
else:
    import pandas as pd
    import numpy as np

    pandas = pd
    numpy = np
    # import scipy

    ACCELERATED = False
