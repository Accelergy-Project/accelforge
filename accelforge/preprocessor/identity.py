from accelforge.frontend import Spec, EinsumName


def add_adapter_between_einsums(
    spec: Spec, einsum_a: EinsumName, einsum_b: EinsumName, adapter
):
    """Insert an adapter between Einsums in a spec, modifying the intermediate tensor names."""
    workload = spec.workload
    e_a = workload.einsums[einsum_a]
    e_b = workload.einsums[einsum_b]

    a2b = e_a.output_tensor_names & e_b.input_tensor_names
    b2a = e_b.output_tensor_names & e_a.input_tensor_names

    if a2b and not b2a:
        producer, consumer = e_a, e_b
        intermediate = next(iter(a2b))
    elif b2a and not a2b:
        producer, consumer = e_b, e_a
        intermediate = next(iter(b2a))
    else:
        raise ValueError(
            f"Cannot insert adapter: expected exactly one intermediate tensor "
            f"flowing between {einsum_a} and {einsum_b}, found a->b={a2b}, b->a={b2a}"
        )

    new_name = f"{intermediate}_{adapter.name}"

    for ta in consumer.tensor_accesses:
        if ta.name == intermediate:
            ta.name = new_name

    adapter_inputs = [ta for ta in adapter.tensor_accesses if not ta.output]
    adapter_outputs = [ta for ta in adapter.tensor_accesses if ta.output]
    if len(adapter_inputs) != 1 or len(adapter_outputs) != 1:
        raise ValueError(
            f"Adapter {adapter.name} must have exactly one input and one output "
            f"tensor access, found {len(adapter_inputs)} inputs and "
            f"{len(adapter_outputs)} outputs"
        )
    adapter_inputs[0].name = intermediate
    adapter_outputs[0].name = new_name

    producer_idx = next(
        i for i, e in enumerate(workload.einsums) if e.name == producer.name
    )
    workload.einsums.insert(producer_idx + 1, adapter)
