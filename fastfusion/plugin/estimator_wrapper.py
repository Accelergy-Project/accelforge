import inspect
import logging
from numbers import Number
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Union
from .estimator import EnergyAreaEstimator
from .logging import move_queue_from_one_logger_to_another, ListLoggable

class PrintableCall:
    def __init__(
        self,
        name: str,
        args: List[str] = (),
        defaults: Dict[str, Any] = None,
    ):
        self.name = name
        self.args = args
        self.defaults = defaults or {}

    def __str__(self):
        n = self.name
        args = [str(a) for a in self.args] + [
            f"{k}={v}" for k, v in self.defaults.items()
        ]

        return f"{n}({', '.join(args)})"

class EnergyAreaQuery:
    """A query to a EnergyArea plug-in."""
    def __init__(
        self,
        class_name: str,
        class_attrs: Dict[str, Any],
        action_name: str = None,
        action_args: Dict[str, Any] = None,
    ):
        self.class_name = class_name
        self.action_name = action_name
        # Attributes and arguments are only included if they are not None
        if action_args is None:
            self.action_args = {}
        else:
            self.action_args = {k: v for k, v in action_args.items() if v is not None}
        self.class_attrs = {k: v for k, v in class_attrs.items() if v is not None}

    def __str__(self):
        attrs_stringified = ", ".join([f"{k}={v}" for k, v in self.class_attrs.items()])
        s = f"{self.class_name}({attrs_stringified})"
        if self.action_name:
            args_stringified = ", ".join(
                [f"{k}={v}" for k, v in self.action_args.items()]
            )
            s += f".{self.action_name}({args_stringified})"
        return s

class Estimation:
    """Estimation object for storing estimation results."""

    def __init__(
        self,
        value: Number,
        success: bool = True,
    ):
        self.value = value
        if not isinstance(value, Number):
            raise TypeError(f"Estimation value must be a number, not {type(value)}")
        self.success = success
        self.messages = []
        self.estimator_name = None

    def add_messages(self, messages: Union[List[str], str]):
        """
        Adds messages to the internal message list. The messages are reported
        depending on plug-in selections and verbosity level.
        """
        if isinstance(messages, str):
            self.add_messages([messages])
        else:
            self.messages += messages

    def __str__(self):
        return f"{self.value}"

    def fail(self, message: str):
        """Marks this estimation as failed and adds the message."""
        if not self.success:
            return
        self.success = False
        self.value = 0
        self.add_messages(message)

    def lastmessage(self) -> str:
        """Returns the last message in the message list. If no messages, returns
        a default."""
        if self.messages:
            return self.messages[-1]
        else:
            return f"No messages found."

    def get_value(self):
        return self.value

class CallableFunction:
    """Wrapper for a function to provide error checking and argument
    matching."""

    def __init__(
        self,
        function: Callable,
        logger: logging.Logger,
        force_name_override: str = None,
        is_init: bool = False,
    ):
        if not isinstance(function, Callable):
            raise TypeError(
                f"Function {function} must be an instance of Callable, not {type(function)}"
            )

        self.function = function
        if is_init:
            function = function.__init__

        args = function.__code__.co_varnames[1 : function.__code__.co_argcount]
        default_length = (
            len(function.__defaults__) if function.__defaults__ is not None else 0
        )

        self.function_name = function.__name__
        if force_name_override is not None:
            self.function_name = force_name_override
        self.non_default_args = args[: len(args) - default_length]
        self.default_args = args[len(args) - default_length :]
        self.default_arg_values = (
            function.__defaults__ if function.__defaults__ is not None else []
        )
        self.logger = logger

    def get_error_message_for_name_match(self, name: str, class_name: str = ""):
        if self.function_name != name:
            return f"Function name {self.function_name} does not match my name {class_name}.{name}"
        return None

    def get_error_message_for_non_default_arg_match(
        self, kwags: dict, class_name: str = ""
    ) -> Optional[str]:
        for arg in self.non_default_args:
            if kwags.get(arg) is None:
                return (
                    f"Argument for {class_name}.{self.function_name} is missing: {arg}. "
                    f'Arguments provided: {", ".join(kwags.keys())}'
                )
        return None

    def get_call_error_message(
        self, name: str, kwargs: dict, class_name: str = ""
    ) -> Optional[str]:
        name_error = self.get_error_message_for_name_match(name, class_name)
        if name_error is not None:
            return name_error
        arg_error = self.get_error_message_for_non_default_arg_match(kwargs, class_name)
        if arg_error is not None:
            return arg_error
        return None

    def call(
        self,
        kwargs: dict,
        class_name: str = "",
        call_function_on_object: object = None,
    ) -> Any:
        kwags_included = {
            k: v
            for k, v in kwargs.items()
            if k in self.non_default_args or k in self.default_args
        }
        unneeded_args = [k for k in kwargs.keys() if k not in kwags_included]
        if unneeded_args:
            self.logger.warn(
                f'Unused arguments ({", ".join(unneeded_args)}) provided for {class_name}.'
                f'{self.function_name}. Arguments used: ({", ".join(kwags_included.keys())})'
            )

        if call_function_on_object is not None:
            return self.function(call_function_on_object, **kwags_included)
        return self.function(**kwags_included)

    def __str__(self):
        return str(
            PrintableCall(
                self.function_name if self.function_name != "__init__" else "",
                self.non_default_args,
                {a: b for a, b in zip(self.default_args, self.default_arg_values)},
            )
        )

class EnergyAreaEstimatorWrapper(ListLoggable):
    """EnergyArea primitive component estimator that wraps a Python class."""

    def __init__(self, estimator_cls: type, class_name: str):
        check_for_valid_estimator_attrs(estimator_cls)
        self.estimator_cls = estimator_cls
        self.estimator_name = class_name
        self.class_name = estimator_cls.name
        super().__init__(name=self.get_name())

        self.percent_accuracy = estimator_cls.percent_accuracy_0_to_100
        self.get_area = CallableFunction(estimator_cls.get_area, self.logger)
        self.leak = CallableFunction(estimator_cls.leak, self.logger)
        self.init_function = CallableFunction(estimator_cls, self.logger, is_init=True)

        self.actions = [
            CallableFunction(getattr(estimator_cls, a), self.logger)
            for a in dir(estimator_cls)
            if getattr(
                getattr(estimator_cls, a),
                "_is_component_energy_action",
                False,
            )
        ]
        self.actions.append(CallableFunction(estimator_cls.leak, self.logger))
        logging.info(
            f"Added estimator {self.estimator_name} that estimates {self.class_name} with actions "
            f'{", ".join(self.get_action_names())}'
        )

    def get_action_names(self) -> List[str]:
        return [a.function_name for a in self.actions]

    def fail_missing(self, missing: str):
        raise AttributeError(
            f"Primitive component {self.class_name} " f"must have {missing}"
        )

    def is_class_supported(self, query: EnergyAreaQuery) -> bool:
        name_check = (
            [self.class_name] if isinstance(self.class_name, str) else self.class_name
        )
        if not query.class_name in name_check:
            self.logger.error(
                f"Class name {query.class_name} is not supported. Supported class "
                f"names: {name_check}"
            )
            return False
        init_error = self.init_function.get_call_error_message(
            "__init__", query.class_attrs, self.class_name
        )
        if init_error is not None:
            self.logger.error(init_error)
            return False
        return True

    def get_initialized_subclass(self, query: EnergyAreaQuery) -> EnergyAreaEstimator:
        subclass = self.init_function.call(query.class_attrs, self.class_name)
        subclass.__ListLoggable__init__()
        return subclass

    def get_matching_actions(self, query: EnergyAreaQuery) -> List[CallableFunction]:
        # Find actions that match the name
        name_matches = [a for a in self.actions if a.function_name == query.action_name]
        if len(name_matches) == 0:
            raise AttributeError(
                f"No action with name {query.action_name} found in {self.class_name}. "
                f'Actions supported: {", ".join(self.get_action_names())}'
            )

        # Find actions that match the arguments
        matching_name_and_arg_actions = [
            a
            for a in name_matches
            if a.get_call_error_message(query.action_name, query.action_args) is None
        ]
        if len(matching_name_and_arg_actions) == 0:
            matching_func_strings = [
                (
                    f"{a.function_name}("
                    + ", ".join(
                        list(a.non_default_args)
                        + ["OPTIONAL " + b for b in a.default_args]
                    )
                )
                + ")"
                for a in name_matches
            ]
            args_provided = (
                query.action_args.keys() if query.action_args else ["<none>"]
            )
            raise AttributeError(
                f"Action with name {query.action_name} found in {self.class_name}, but provided "
                f"arguments do not match.\n\t"
                f'Arguments provided: {", ".join(args_provided)}\n\t'
                f"Possible actions:\n\t\t" + "\n\t\t".join(matching_func_strings)
            )
        return matching_name_and_arg_actions

    def estimate_energy(self, query: EnergyAreaQuery) -> Estimation:
        """Returns the energy estimation for the given action."""
        initialized_obj = self.get_initialized_subclass(query)
        move_queue_from_one_logger_to_another(initialized_obj.logger, self.logger)
        supported_actions = self.get_matching_actions(query)
        if len(supported_actions) == 0:
            raise AttributeError(
                f"No action with name {query.action_name} found in {self.class_name}. "
                f'Actions supported: {", ".join(self.get_action_names())}'
            )
        try:
            estimation = Estimation(
                value=supported_actions[0].call(
                    query.action_args, self.class_name, initialized_obj
                )
            )
        except Exception as e:
            move_queue_from_one_logger_to_another(initialized_obj.logger, self.logger)
            raise e
        move_queue_from_one_logger_to_another(initialized_obj.logger, self.logger)
        return estimation

    def estimate_area(self, query: EnergyAreaQuery) -> Estimation:
        """Returns the area estimation for the given action."""
        return Estimation(value=self.get_initialized_subclass(query).get_area())

    def get_name(self) -> str:
        return self.estimator_name

    def get_class_names(self) -> List[str]:
        return (
            [self.class_name] if isinstance(self.class_name, str) else self.class_name
        )

    @staticmethod
    def print_action(action: CallableFunction) -> str:
        return action.function_name


def check_for_valid_estimator_attrs(estimator: EnergyAreaEstimator):
    # Check for valid class_name. Must be a string or list of strings
    if getattr(estimator, "name", None) is None:
        raise AttributeError(f"EnergyAreaEstimator {estimator} must have a name attribute")
    name = estimator.name
    if not isinstance(name, str) and not (
        isinstance(name, list) and all(isinstance(n, str) for n in name)
    ):
        raise AttributeError(
            f"EnergyAreaEstimator {estimator} class_name must be a string or list of strings"
        )

    # Check for valid percent_accuracy. Must be a number between 0 and 100
    if getattr(estimator, "percent_accuracy_0_to_100", None) is None:
        raise AttributeError(
            f'EnergyAreaEstimator for {name} must have a "percent_accuracy_0_to_100" '
            f"attribute."
        )
    percent_accuracy = estimator.percent_accuracy_0_to_100
    if not isinstance(percent_accuracy, Number):
        raise AttributeError(
            f"EnergyAreaEstimator for {name} percent_accuracy_0_to_100 must be a "
            f"number. It is currently a {type(percent_accuracy)}"
        )
    if percent_accuracy < 0 or percent_accuracy > 100:
        raise AttributeError(
            f"EnergyAreaEstimator for {name} percent_accuracy_0_to_100 must be "
            f"between 0 and 100 inclusive."
        )
