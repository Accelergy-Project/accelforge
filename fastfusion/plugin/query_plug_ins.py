import logging
import copy
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, Union
from .logging import get_logger, pop_all_messages, log_all_lines, clear_logs
from .estimator_wrapper import EnergyAreaEstimatorWrapper, EnergyAreaQuery, Estimation

RAISED_WARNINGS_FOR_CLASSES = []


def indent_list_text_block(prefix: str, list_to_print: List[str]):
    if not list_to_print:
        return ""
    return "\n| ".join(
        [f"{prefix}"] + [str(l).replace("\n", "\n|  ") for l in list_to_print]
    )


def call_plug_in(
    plug_in: EnergyAreaEstimatorWrapper,
    query: EnergyAreaQuery,
    target_func: Callable,
) -> Estimation:
    # Clear the logger
    pop_all_messages(plug_in.logger)
    try:
        estimation = target_func(query)
    except Exception as e:
        estimation = Estimation(0, success=False)
        plug_in.logger.error(f"{type(e).__name__}: {e}")

    # Add message logs
    estimation.add_messages(pop_all_messages(plug_in.logger))
    estimation.estimator_name = plug_in.get_name()

    # See if this estimation matches user requested plug-in and min accuracy
    attrs = query.class_attrs
    if (
        attrs.get("plug_in", None) is not None
        and attrs["plug_in"] != estimation.estimator_name
    ) or (
        attrs.get("min_accuracy", None) is not None
        and attrs["min_accuracy"] > estimation.percent_accuracy_0_to_100
    ):
        estimation.fail(
            f"Plug-in {estimation.estimator_name} was not selected for query."
        )
    return estimation


def get_energy_estimation(plug_in: Any, query: EnergyAreaQuery) -> Estimation:
    e = call_plug_in(plug_in, query, plug_in.estimate_energy)
    if e and e.success and query.action_name == "leak":
        n_instances = query.class_attrs.get("n_instances", 1)
        e.add_messages(f"Multiplying by n_instances {n_instances}")
        e.value *= n_instances
    return e


def get_area_estimation(plug_in: Any, query: EnergyAreaQuery) -> Estimation:
    e = call_plug_in(plug_in, query, plug_in.estimate_area)
    if e and e.success:
        n_instances = query.class_attrs.get("n_instances", 1)
        e.add_messages(f"Multiplying by n_instances {n_instances}")
        e.value *= n_instances
    return e


def get_best_estimate(
    models: List[EnergyAreaEstimatorWrapper],
    query: EnergyAreaQuery,
    is_energy_estimation: bool,
) -> Estimation:
    est_func = get_energy_estimation if is_energy_estimation else get_area_estimation

    target = "ENERGY" if is_energy_estimation else "AREA"
    if logging.getLogger("").isEnabledFor(logging.INFO):
        logging.getLogger("").info("")
    logging.getLogger("").info(f"{target} ESTIMATION for {query}")

    for to_drop in ["area", "energy", "area_scale", "energy_scale"]:
        for drop_from in [query.class_attrs, query.action_args]:
            if to_drop in drop_from:
                del drop_from[to_drop]

    estimations = []
    supported_models = sorted(
        models, key=lambda x: x.percent_accuracy, reverse=True
    )
    supported_models = [p for p in supported_models if p.is_class_supported(query)]

    if not supported_models:
        if not models:
            raise KeyError(f"No plug-ins found. Please check your configuration.")
        supported_classes = set.union(*[set(p.get_class_names()) for p in models])
        raise KeyError(
            f"Class {query.class_name} is not supported by any plug-ins. Supported classes: {supported_classes}"
        )

    estimation = None
    for plug_in in supported_models:
        estimation = est_func(plug_in, copy.deepcopy(query))
        logger = get_logger(plug_in.get_name())
        if not estimation.success:
            estimation.add_messages(pop_all_messages(logger))
            estimations.append((plug_in.percent_accuracy, estimation))
        else:
            log_all_lines(
                f"EnergyArea",
                "info",
                f"{estimation.estimator_name} estimated "
                f"{estimation} with accuracy {plug_in.percent_accuracy}. "
                + indent_list_text_block("Messages:", estimation.messages),
            )
            break

    full_logs = [
        indent_list_text_block(
            f"{e.estimator_name} with accuracy {a} estimating value: ", e.messages
        )
        for a, e in estimations
    ]
    fail_reasons = [
        f"{e.estimator_name} with accuracy {a} estimating value: " f"{e.lastmessage()}"
        for a, e in estimations
    ]

    if full_logs:
        log_all_lines(
            "EnergyArea", "debug", indent_list_text_block("Estimator logs:", full_logs)
        )
    if fail_reasons:
        log_all_lines(
            "EnergyArea",
            "debug",
            indent_list_text_block("Why plug-ins did not estimate:", fail_reasons),
        )
    if fail_reasons:
        log_all_lines(
            "EnergyArea",
            "info",
            indent_list_text_block(
                "Plug-ins provided accuracy, but failed to estimate:",
                fail_reasons,
            ),
        )

    if estimation and estimation.success:
        return estimation

    clear_logs()

    estimation_target = "energy" if is_energy_estimation else "area"
    raise RuntimeError(
        f"Can not find an {estimation_target} estimator for {query}\n"
        f'{indent_list_text_block("Logs for plug-ins that could estimate query:", full_logs)}\n'
        f'{indent_list_text_block("Why plug-ins did not estimate:", fail_reasons)}\n'
        f'\n.\n.\nTo see a list of available component models, run "<command you used> -h" and '
        f"find the option to list components. Alternatively, run accelergy verbose and "
        f"check the log file."
    )
