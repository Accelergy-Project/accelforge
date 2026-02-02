import os

DUMP_ISL_IR: bool = os.getenv("ACCELFORGE_DUMP_ISL_IR") == "1"
LOG_ISL_IR: bool = os.getenv("ACCELFORGE_LOG_ISL_IR") == "1"
