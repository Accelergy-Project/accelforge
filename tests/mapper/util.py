from pathlib import Path
import islpy as isl

TEST_CONFIG_PATH: Path = Path(__file__).parent / "configs"

def to_isl_maps(obj) -> dict:
    def _to_isl_maps(obj):
        """Recursively convert string ISL maps to isl.Map; leave others alone."""
        if isinstance(obj, str):
            return isl.Map.read_from_str(isl.DEFAULT_CONTEXT, obj)
        if isinstance(obj, dict):
            return {k: _to_isl_maps(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_isl_maps(v) for v in obj]
        return obj
    return _to_isl_maps(obj) # type: ignore