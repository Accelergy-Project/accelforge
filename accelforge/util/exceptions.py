"""Public exceptions for AccelForge."""

from typing import Any


class EvaluationError(Exception):
    """Exception raised when parsing fails.

    This exception is raised when there's an error parsing specifications,
    architectures, workloads, or mappings.

    Parameters
    ----------
    *args
        Standard exception arguments.
    source_field : Any, optional
        The field where the error occurred.
    message : str, optional
        Error message describing what went wrong.
    **kwargs
        Additional keyword arguments.
    """

    def __init__(self, *args, source_field: Any = None, message: str = None, **kwargs):
        self._fields = [source_field] if source_field is not None else []
        if message is None and len(args) > 0:
            message = args[0]
        self.message = message
        super().__init__(*args, **kwargs)

    def add_field(self, field: Any):
        """
        Add a field to the error context. The output error message will include the
        field path as a period-separated string like "spec.arch.nodes.0.name".

        Parameters
        ----------
        field : Any
            The field to add to the error context.
        """
        self._fields.append(field)

    def __str__(self) -> str:
        s = f"{self.__class__.__name__} in {'.'.join(str(field) for field in self._fields[::-1])}"
        if self.message is not None:
            s += f": {self.message}"
        if getattr(self, "__notes__", None):
            s += "\n" + "\n".join(self.__notes__)
        return s
