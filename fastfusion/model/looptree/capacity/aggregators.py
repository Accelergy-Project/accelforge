def compute_max(child_caps, caps):
    buf_to_max = {}
    for child_cap in child_caps:
        buf_to_child_usage = {}

        for buf, v in child_cap.items():
            if buf not in buf_to_child_usage:
                buf_to_child_usage[buf] = 0
            if buf not in buf_to_max:
                buf_to_max[buf] = 0

            buf_to_child_usage[buf] += v

        for buf in buf_to_child_usage:
            buf_to_max[buf] = max(buf_to_max[buf], buf_to_child_usage[buf])

    for buf, c in buf_to_max.items():
        if buf not in caps:
            caps[buf] = 0
        caps[buf] += c
        


def compute_total(child_caps, caps):
    for child_cap in child_caps:
        for buf, v in child_cap.items():
            if buf in caps:
                caps[buf] += v
            else:
                caps[buf] = v


CAPACITY_AGGREGATORS = {
    'sequential': compute_max,
    'pipeline': compute_total,
    'parallel': compute_total
}