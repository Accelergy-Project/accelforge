def gather_relevant_boundary_idxs(ranks, relevant_ranks):
    idxs = []
    last_is_relevant = True
    for i, r in enumerate(ranks):
        is_relevant = r in relevant_ranks
        if last_is_relevant and not is_relevant:
            idxs.append(i)
        last_is_relevant = is_relevant
    if last_is_relevant:
        idxs.append(len(ranks))
    return idxs