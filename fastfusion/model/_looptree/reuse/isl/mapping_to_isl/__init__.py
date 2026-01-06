import os

DUMP_ISL_IR: bool = os.getenv("FASTFUSION_DUMP_ISL_IR") == "1"
LOG_ISL_IR: bool = os.getenv("FASTFUSION_LOG_ISL_IR") == "1"
