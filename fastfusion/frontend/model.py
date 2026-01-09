from fastfusion.frontend.mapper.metrics import Metrics
from fastfusion.util._basetypes import ParsableModel


class Model(ParsableModel):
    """Configuration for the model."""

    metrics: Metrics = Metrics.all_metrics()
    """
    Metrics to evaluate.
 
    If using spec to call mapper, leave this configuration as is. The mapper 
    will make necessary configurations.
    """