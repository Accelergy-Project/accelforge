""" Base class for all processors."""

from abc import abstractmethod, ABC
import copy
import logging
from .nodes import Node
from typing import Any, Optional


class Processor(ABC):
    """Base class for all processors, which are used to modify the specification before it is run.

    Attributes:
        spec: The specification to process.
        logger: The logger for this processor.
    """

    def __init__(self, spec: Optional["Specification"] = None):
        self._initialized: bool = True
        self.logger = logging.getLogger(self.__class__.__name__)

    def pre_parse_process(self, spec: "Specification"):
        """Process the specification before parsing."""
        self.logger.debug(f"Pre-parse processing with {self}")
        self.spec = spec

    @abstractmethod
    def process(self, spec: "Specification"):
        """Process the specification."""
        self.logger.debug(f"Processing with {self}")
        self.spec = spec

    def declare_attrs(self):
        """
        Initialize the attributes that the processor is responsible for.
        !@note This method is called before process() is called. See the
               SimpleProcessor for an example.
        """
        pass

    def get_index(self, processor_type: type, spec: "Specification"):
        """Get the index of the processor in the list of processors."""
        for i, processor in enumerate(spec._processors_run):
            if isinstance(processor, type):
                return i
            if isinstance(processor, processor_type):
                return i
            if processor == processor_type:
                return i
        return -1

    def must_run_after(
        self, other: type, spec: "Specification", ok_if_not_found: bool = False
    ):
        """Ensure that this processor runs after another processor.
        !@param other: The processor that this processor must run after.

        !@param ok_if_not_found: If False, OK if the other processor is not
                                 found. If True, raise an exception if the
                                 other processor is not found.
        """
        other_idx = self.get_index(other, spec)
        my_idx = self.get_index(self.__class__, spec)
        if other_idx > my_idx or (other_idx == -1 and not ok_if_not_found):
            raise ProcessorError(
                f"{other.__name__} must run before {self.__class__.__name__}. "
                f"Please add {other.__name__} to the list of processors "
                f"before {self.__class__.__name__} in the spec."
            )

    def add_attr(self, target: Node, *args, **kwargs):
        # ensure target is a class
        if not isinstance(target, type):
            raise TypeError(
                f"Can not add attribute to a class instance, only to a class. "
            )
        if not issubclass(target, Node):
            raise TypeError(f"Can only add attributes to Node subclasses ")

        target.add_attr(
            *args,
            **kwargs,
            _processor_responsible_for_removing=self,
            _add_checker_to=self.spec._processor_attributes.setdefault(
                target.unique_class_name(), {}
            ),
        )


class SimpleProcessor(Processor):
    """An example simple processor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger.info("Initializing SimpleProcessor")

    def declare_attrs(self):
        """Initialize the attributes that the processor handles."""
        from ..v4 import Problem

        super().add_attr(Problem, "simple_processor_attr", str, "")

    def process(self, spec: "Specification"):
        """Process the specification. Remove attributes that this
        processor is responsible for."""
        if "simple_processor_attr" in spec.problem:
            del spec.problem["simple_processor_attr"]
            self.logger.info('Deleted "simple_processor_attr"')


def set_parents(n: Node):
    for _, x in n.items():
        if isinstance(x, Node):
            x.parent_node = n


class References2CopiesProcessor(Processor):
    """Converts references to copies in the specification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, spec: "BaseSpecification"):
        super().process(spec)
        seen_ids = set()
        visited = []
        processors = spec.processors  # Don't copy processors
        spec.processors = []
        self.refs2copies_fast(spec, spec, seen_ids, visited)
        spec.processors = processors

    def refs2copies_fast(
        self, spec: "BaseSpecification", n: Any, seen_ids=None, visited=None, depth=0
    ) -> Any:
        visited.append(n)  # Avoid garbage collection
        if isinstance(n, Node):
            n.parent_node = None

        if id(n) in seen_ids:
            n = copy.deepcopy(n)
        seen_ids.add(id(n))

        if not isinstance(n, Node):
            return n

        if isinstance(n, Node):
            for i, x in n.items():
                # self.logger.debug(
                #     "Depth %s Copying %s in %s[%s]", depth, str(x), str(n), i
                # )  # type: ignore
                n[i] = self.refs2copies_fast(spec, x, seen_ids, visited, depth + 1)
                if isinstance(n[i], Node):
                    n[i].parent_node = n
                    n[i].spec = spec
            n.parent_node = None
        return n


class ProcessorError(Exception):
    """Exception raised by processors."""
