import copy
import time
from typing import Any, Dict, List, Optional, Union
from .nodes import DictNode, ListNode, Node
from .processor import Processor, ProcessorError, References2CopiesProcessor
from .parse_expressions import ParseExpressionsContext


def class2obj(x):
    return x() if isinstance(x, type) else x


class ProcessorListHolder(ListNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("", callfunc=class2obj)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return super().__getitem__(key)


class BaseSpecification(DictNode):
    """
    Base class for specifications in the Timeloop framework.

    Attributes:
        processors (ListNode): List of processors associated with the specification.
        _required_processors (ListNode): List of required processors.
        _parsed_expressions (bool): Flag indicating whether expressions have been parsed.
        _processors_run (List[Processor]): List of processors that have been run.
    """

    @classmethod
    def declare_attrs(cls, *args, **kwargs): ...
class BaseSpecification(DictNode):
    @classmethod
    def declare_attrs(cls, *args, **kwargs):
        super().declare_attrs(*args, **kwargs)
        super().add_attr("ignore", part_name_match=True, no_change_key=True)
        super().add_attr("processors", ListNode, [])
        super().add_attr("_required_processors", ListNode, [])
        super().add_attr("_parsed_expressions", bool, False)
        super().add_attr("_processors_run", ListNode, [])
        super().add_attr("_processors_run_pre_parse", ListNode, [])

    def _claim_nodes(self, *args, **kwargs):
        def claim_node(n: Node):
            if isinstance(n, Node):
                n.spec = self

        self.recursive_apply(claim_node)

    def _processors_declare_attrs(self, *args, **kwargs):
        Node.reset_processor_elems()
        for p in self.processors + self._required_processors:
            p.spec = self  # MAKE SURE THIS IS KEPT UP TO DATE
            p.declare_attrs()

    def _early_init_processors(self, _required_processors: List["Processor"], **kwargs):
        kwargs.setdefault("processors", [])
        kwargs["_required_processors"] = _required_processors

        self.processors = ProcessorListHolder(kwargs["processors"])
        self._required_processors = ProcessorListHolder(kwargs["_required_processors"])
        self._processors_declare_attrs()

    def __init__(self, *args, **kwargs):
        self._processor_attributes = {}
        Node.set_global_spec(self)
        self.spec = self

        self._early_init_processors(**kwargs)  # Because processors define declare_attrs

        super().__init__(*args, **kwargs)

        self.processors: ListNode[Processor] = self["processors"]
        self._required_processors: ListNode[Processor] = self["_required_processors"]
        self._processors_run: List[Processor] = self["_processors_run"]
        self._processors_run_pre_parse: List[Processor] = self[
            "_processors_run_pre_parse"
        ]
        self._parsed_expressions = self["_parsed_expressions"]
        self.process(References2CopiesProcessor, check_types=False)

    def needs_processing(
        self,
        with_processors: Optional[List["Processor"]] = None,
        to_run: Optional[List["Processor"]] = None,
        pre_parse: bool = False,
    ):
        if with_processors is None:
            with_processors = self.processors + [References2CopiesProcessor]
        to_check = (to_run or []) + (
            self._processors_run_pre_parse if pre_parse else self._processors_run
        )

        for p in with_processors:
            for x in to_check:
                # If p is a class, check if x is an instance of that class
                if isinstance(p, type) and isinstance(x, p):
                    break
                elif isinstance(x, type) and isinstance(p, x):
                    break
                elif p == x:
                    break
            else:
                return True
        return False

    def process(
        self,
        with_processors: Union["Processor", List["Processor"]] = None,
        check_types: bool = False,
        check_types_ignore_empty: bool = True,
        reprocess: bool = True,
    ):
        """
        Process the specification with the given processors.

        Args:
            with_processors (Union[Processor, List[Processor]], optional): Processors to be used for processing the specification. Defaults to None.
            check_types (bool, optional): Flag indicating whether to check for unrecognized types. Defaults to False.
            check_types_ignore_empty (bool, optional): Flag indicating whether to ignore empty types during type checking. Defaults to True.
            reprocess (bool, optional): Flag indicating whether to reprocess the specification even if it has been processed before. Defaults to True.
        """
        prev_global_spec = Node.get_global_spec()
        try:
            Node.set_global_spec(self)
            if with_processors is None:
                processors = self.processors
            else:
                if not isinstance(with_processors, (list, tuple)):
                    with_processors = [with_processors]
                processors = [p for p in with_processors]

            if self.needs_processing([References2CopiesProcessor], processors):
                self.process(References2CopiesProcessor, check_types=False)

            overall_start_time = time.time()
            for i, p in enumerate(processors):
                if not self.needs_processing([p]) and (
                    not reprocess
                    or p == References2CopiesProcessor
                    or isinstance(p, References2CopiesProcessor)
                ):
                    continue
                # If the processor isn't initialized, initialize it
                p_cls = p
                p = class2obj(p)
                Node.reset_processor_elems(p.__class__)
                processors[i] = p
                self.logger.info("Running processor %s", p.__class__.__name__)
                start_time = time.time()
                p.process(self)
                self.logger.info(
                    "Processor %s done after %.2f seconds",
                    p.__class__.__name__,
                    time.time() - start_time,
                )
                self._processors_run.append(p_cls)
            if check_types:
                self.check_unrecognized(ignore_empty=check_types_ignore_empty)
            self.logger.info(
                "Specification processed in %.2f seconds",
                time.time() - overall_start_time,
            )
        finally:
            Node.set_global_spec(prev_global_spec)

    def parse_expressions(
        self,
        symbol_table: Optional[Dict[str, Any]] = None,
        parsed_ids: Optional[set] = None,
    ):
        """
        Parse expressions in the specification.

        Args:
            symbol_table (Optional[Dict[str, Any]], optional): Symbol table to be used for parsing. Defaults to None.
            parsed_ids (Optional[set], optional): Set of IDs of specifications that have already been parsed. Defaults to None.
        """
        if self.needs_processing([References2CopiesProcessor]):
            raise ProcessorError(
                f"Must run References2CopiesProcessor before "
                f"parsing expressions. Call process() with "
                f"any arguments."
            )
        for p in self.processors:
            if self.needs_processing([p], pre_parse=True):
                class2obj(p).pre_parse_process(self)
                self._processors_run_pre_parse.append(p)

        with ParseExpressionsContext(self):
            symbol_table = {} if symbol_table is None else symbol_table.copy()
            parsed_ids = set() if parsed_ids is None else parsed_ids
            parsed_ids.add(id(self))
            symbol_table["spec"] = self
            super().parse_expressions(symbol_table, parsed_ids)
            self.check_unrecognized(ignore_should_have_been_removed_by=1)
            self._parsed_expressions = True

    def _process(self):
        spec = copy.deepcopy(self)
        if not spec._parsed_expressions:
            spec.parse_expressions()
        if spec.needs_processing():
            spec.process(check_types=False, reprocess=False)
        spec.process(spec._required_processors)
        spec.check_unrecognized()
        return spec

    def __deepcopy__(self, memo):
        new_instance = super().__deepcopy__(memo)
        new_instance._claim_nodes()
        return new_instance
