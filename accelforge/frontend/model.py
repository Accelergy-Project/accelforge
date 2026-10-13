from accelforge.frontend.mapper.metrics import Metrics
from accelforge.util._basetypes import EvalableModel


class Model(EvalableModel):
    """Configuration for the model."""

    metrics: Metrics = Metrics.all_metrics()
    """
    Metrics to evaluate.

    If using spec to call mapper, leave this configuration as is. The mapper
    will make necessary configurations.
    """

    _use_new_latency_model: bool = False

    def __init__(self, **kwargs):
        use_new_latency_model = kwargs.pop("_use_new_latency_model", False)
        super().__init__(**kwargs)
        self._use_new_latency_model = use_new_latency_model
