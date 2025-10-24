import sys, os
sys.path.append(os.getcwd() + "/src")
from re import L
from tile_graph import GenericTileGraph
from functools import reduce
import copy
import src

MAC_ENERGY = 0.07e-12  # 0.07pJ per MAC

def enumerate_spatial(prob, tile_to_unroll, unroll_to_try, x_axis, y_axis, nd=True, max_util=True):

    DBG = False
    if x_axis == 4 and y_axis == 4:
        DBG = True

    xmemo = {}
    ret = []
    sp_memo = [{} for _ in prob.tens_desc]

    run_num = 0
    while len(ret) == 0:
        run_num += 1
        for sp, tens_id in unroll_to_try:
            xgraph = GenericTileGraph(tile_to_unroll, None, None)
            if sp[0] not in xmemo:
                xmemo[sp[0]] = xgraph.bottom_up_spatial(x_axis, sp[0], nd=nd, max_util=max_util)

            spxs = xmemo[sp[0]]

            if not spxs and run_num == 1:
                continue

            if not spxs:
                if DBG:
                    print(
                        "came here for", tile_to_unroll, "   and ", sp, " - ", tens_id
                    )
                spxs = []
                # try finding only reuse bounds, but does not need to use all PEs
                full_e = xgraph.bottom_up_spatial(
                    x_axis, sp[0], nd=True, max_util=False
                )
                spxs.extend(full_e)
                min_occ = None
                for e in full_e:
                    occ = 1
                    for v in e.values():
                        occ *= v
                    if min_occ is None or occ < min_occ:
                        min_occ = occ
                # cannot fully unroll with just the reuse bounds
                # try allowing other bounds (but with majority still with reuse bounds)
                max_util_mixed = xgraph.bottom_up_spatial(
                    x_axis, "".join(prob.bounds.keys()), nd=True, prior=sp[0]
                )
                if max_util_mixed:
                    spxs.extend(max_util_mixed)
                else:
                    # cannot find anything, try anything at this point
                    mixed = xgraph.bottom_up_spatial(
                        x_axis,
                        "".join(prob.bounds.keys()),
                        nd=True,
                        max_util=False,
                        min_occ=min_occ,
                    )
                    spxs.extend(mixed)

            for spx in spxs:
                tile_to_unroll_y = tile_to_unroll.copy()
                for k, v in spx.items():
                    tile_to_unroll_y[k] = int(tile_to_unroll_y[k] / v)
                ygraph = GenericTileGraph(tile_to_unroll_y, None, None)
                spys = ygraph.bottom_up_spatial(y_axis, sp[1], nd=nd, max_util=max_util)

                if not spys:
                    spys = []
                    # try finding only reuse bounds, but does not need to use all PEs
                    full_e = ygraph.bottom_up_spatial(
                        y_axis, sp[1], nd=True, max_util=False
                    )
                    spys.extend(full_e)

                    min_occ = None
                    for e in full_e:
                        occ = 1
                        for v in e.values():
                            occ *= v
                        if min_occ is None or occ < min_occ:
                            min_occ = occ

                    # cannot fully unroll with just the reuse bounds
                    # try allowing other bounds (but with majority still with reuse bounds)

                    max_util_mixed = ygraph.bottom_up_spatial(
                        y_axis, "".join(prob.bounds.keys()), nd=True, prior=sp[1]
                    )
                    if max_util_mixed:
                        spys.extend(max_util_mixed)
                    else:
                        # cannot find anything, try anything at this point (that is still better than the worse energy occ)
                        mixed = ygraph.bottom_up_spatial(
                            y_axis,
                            "".join(prob.bounds.keys()),
                            nd=True,
                            max_util=False,
                            min_occ=min_occ,
                        )
                        spys.extend(mixed)

                for spy in spys:
                    sp_key = tuple(list(spx.values()) + list(spy.values()))
                    sp_key = tuple(
                        [x[0] * x[1] for x in zip(spx.values(), spy.values())]
                    )
                    if sp_key not in sp_memo[tens_id]:
                        sp_memo[tens_id][sp_key] = True
                        ret.append((spx, spy, tens_id))
    return ret


# ****************************** Unit Tile ******************************
# 1- Function returns the smallest possible problem (all ones), often used for initialization
def smallest_prob(prob):
    return dict([(x, 1) for x in prob.bounds.keys()])


def enumerate_tiles(
    tiles, prob, mem_size, static=False, prior=False, banks=None, bypass=None
):
    ret = []
    for tile, order in tiles:
        if len(tile.subtiles) > 1:
            small_prob = tile.get_dual_factors()[-1][0]
        else:
            small_prob = smallest_prob()

        subtile_graph = GenericTileGraph(
            tile.subtiles[-1],
            small_prob,
            prob.tens_desc,
            size_fun=prob.size_fun,
        )

        if static:
            prior_arg = order[-1] if prior else None
            for pot_tile in subtile_graph.bottom_up_static(
                mem_size,
                static=list(prob.static),
                prior=prior_arg,
                banks=banks,
            ):
                ret.append((tile.split(pot_tile, tail=True), order))
        else:
            prior_arg = order[-1] if prior else None
            for pot_tile in subtile_graph.bottom_up(
                mem_size, prior_arg, banks=banks, bypass=bypass
            ):
                ret.append((tile.split(pot_tile, tail=True), order))
    return ret


# ****************************** Cost Function ******************************
# - According to access_all (# of accesses per level), finds best ordering
#   per level (EDP-wise)
def get_cost_edp(prob, accesses_all, access_energies, bw, active_pes=1, spec=True):
    src.counter.counter += 1

    # product of all problem factors
    macs = 1
    for _, v in prob.bounds.items():
        macs *= v
    bl_lat = macs / active_pes
    best_lat = bl_lat

    # energy of L1 reads & MAC
    mac_e = macs * MAC_ENERGY
    accesses, spec_accesses = accesses_all
    op_accesses = prob.get_operand_accesses()  # L1 ifmap, w, ofmap read & ofmap update
    op_access_e = 0
    for i, access in enumerate(
        op_accesses[:-1]
    ):  # last 1 is ofmap (both read & update)
        op_access_e += access * access_energies[0][i]
    op_access_e += (op_accesses[-1][0] + op_accesses[-1][1]) * access_energies[0][-1]
    energy = mac_e + op_access_e

    # for each level, find the best order (EDP wise) based on tiling (which gives access numbers)
    ordering = []
    for lev, access in enumerate(accesses):
        loc_best_cost = None
        loc_best_e = None
        loc_best_lat = None
        loc_best_ord = None
        ord_to_try = access.keys()  # access is a dict of accesses for each order
        # print("\n\n\naccess keys is", ord_to_try)

        # calculate total access energy and bw for each order in this level
        for k in ord_to_try:
            v = access[k]
            loc_e = 0
            loc_lat = bl_lat
            tot_acc_for_rbw = 0  # for read bw
            tot_acc_for_wbw = 0  # for write bw
            for i, vv in enumerate(v[:-1]):  # for tensor operands except ofmap
                loc_e += access_energies[lev][i] * vv[0]  # current level
                loc_e += access_energies[lev + 1][i] * vv[1]  # level above it
                tot_acc_for_rbw += vv[1]
            ofmp_wrt = v[-1][0][1]
            ofmp_updt = v[-1][1][0]
            ofmap_rd = v[-1][1][1]
            loc_e += access_energies[lev][-1] * ofmp_wrt
            loc_e += access_energies[lev + 1][-1] * ofmap_rd
            loc_e += access_energies[lev + 1][-1] * ofmp_updt
            tot_acc_for_rbw += ofmap_rd
            tot_acc_for_wbw += ofmp_updt

            # calculate NoC throttle and cost
            # Currently only supports spatial levels, update to detect throttling too
            if bw[lev + 1]:
                loc_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    loc_lat,
                )
            loc_cost = (energy + loc_e) * loc_lat

            # update best order found so far
            if (loc_best_cost is None) or (loc_cost < loc_best_cost):
                loc_best_cost = loc_cost
                loc_best_e = loc_e
                loc_best_lat = loc_lat
                loc_best_ord = k

        ordering.append(loc_best_ord)
        energy += loc_best_e
        # update baseline/default latency
        bl_lat = max(bl_lat, loc_best_lat)
    spec_best_cost = None
    for order, access in spec_accesses.items():
        spec_e = mac_e + op_access_e
        spec_lat = best_lat

        for lev, acc in enumerate(access):
            tot_acc_for_rbw = 0
            tot_acc_for_wbw = 0
            for i, vv in enumerate(acc[:-1]):
                spec_e += access_energies[lev][i] * vv[0]
                spec_e += access_energies[lev + 1][i] * vv[1]
                tot_acc_for_rbw += vv[1]
            spec_e += access_energies[lev][-1] * acc[-1][0][1]
            spec_e += access_energies[lev + 1][-1] * acc[-1][1][1]
            spec_e += access_energies[lev + 1][-1] * acc[-1][1][0]
            tot_acc_for_wbw += acc[-1][1][0]
            tot_acc_for_rbw += acc[-1][1][1]
            if bw[lev + 1]:
                spec_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    spec_lat,
                )

        if (spec_best_cost is None) or ((spec_e * spec_lat) < spec_best_cost):
            spec_best_cost = spec_e * spec_lat
            spec_best_cycle = spec_lat
            spec_best_e = spec_e
            spec_best_ord = list(order)
    if spec_best_cost and spec_best_cost < (energy * bl_lat) and spec:
        return spec_best_cost, spec_best_ord, spec_best_cycle, spec_best_e
    return energy * bl_lat, ordering, bl_lat, energy


def get_cost_edp_dr(prob, accesses_all, access_energies, bw, active_pes=1, spec=True):
    src.counter.counter += 1

    # product of all problem factors
    macs = 1
    for _, v in prob.bounds.items():
        macs *= v
    bl_lat = macs / active_pes
    best_lat = bl_lat

    # energy of L1 reads & MAC
    mac_e = macs * MAC_ENERGY
    accesses, spec_accesses = accesses_all
    op_accesses = prob.get_operand_accesses()  # L1 ifmap, w, ofmap read & ofmap update
    op_access_e = 0
    for i, access in enumerate(
        op_accesses[:-1]
    ):  # last 1 is ofmap (both read & update)
        op_access_e += access * access_energies[0][i]
    op_access_e += (op_accesses[-1][0] + op_accesses[-1][1]) * access_energies[0][-1]
    energy = mac_e + op_access_e
    dr_e = 0
    # for each level, find the best order (EDP wise) based on tiling (which gives access numbers)
    ordering = []
    for lev, access in enumerate(accesses):

        loc_best_dr_cst = None
        loc_best_cost = None
        loc_best_e = None
        loc_best_lat = None
        loc_best_ord = None
        ord_to_try = access.keys()  # access is a dict of accesses for each order
        # print("\n\n\naccess keys is", ord_to_try)

        # calculate total access energy and bw for each order in this level
        for k in ord_to_try:
            v = access[k]
            lc_dr_e = 0
            loc_e = 0
            loc_lat = bl_lat
            tot_acc_for_rbw = 0  # for read bw
            tot_acc_for_wbw = 0  # for write bw
            for i, vv in enumerate(v[:-1]):  # for tensor operands except ofmap
                loc_e += access_energies[lev][i] * vv[0]  # current level
                loc_e += access_energies[lev + 1][i] * vv[1]  # level above it
                lc_dr_e += access_energies[lev][i] * vv[0]  # current level
                lc_dr_e += access_energies[lev + 1][i] * vv[1]
                tot_acc_for_rbw += vv[1]
            ofmp_wrt = v[-1][0][1]
            ofmp_updt = v[-1][1][0]
            ofmap_rd = v[-1][1][1]
            loc_e += access_energies[lev][-1] * ofmp_wrt
            loc_e += access_energies[lev + 1][-1] * ofmap_rd
            loc_e += access_energies[lev + 1][-1] * ofmp_updt

            lc_dr_e += access_energies[lev][-1] * ofmp_wrt
            lc_dr_e += access_energies[lev + 1][-1] * ofmap_rd
            lc_dr_e += access_energies[lev + 1][-1] * ofmp_updt

            tot_acc_for_rbw += ofmap_rd
            tot_acc_for_wbw += ofmp_updt

            # calculate NoC throttle and cost
            # Currently only supports spatial levels, update to detect throttling too
            if bw[lev + 1]:
                loc_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    loc_lat,
                )
            loc_cost = (energy + loc_e) * loc_lat

            # update best order found so far
            if (loc_best_cost is None) or (loc_cost < loc_best_cost):
                loc_best_cost = loc_cost
                loc_best_e = loc_e
                loc_best_lat = loc_lat
                loc_best_ord = k

                loc_best_dr_cst = lc_dr_e

        ordering.append(loc_best_ord)
        energy += loc_best_e
        dr_e += loc_best_dr_cst
        # update baseline/default latency
        bl_lat = max(bl_lat, loc_best_lat)
    spec_best_cost = None
    for order, access in spec_accesses.items():
        spec_e = mac_e + op_access_e
        spec_lat = best_lat

        for lev, acc in enumerate(access):
            tot_acc_for_rbw = 0
            tot_acc_for_wbw = 0
            for i, vv in enumerate(acc[:-1]):
                spec_e += access_energies[lev][i] * vv[0]
                spec_e += access_energies[lev + 1][i] * vv[1]
                tot_acc_for_rbw += vv[1]
            spec_e += access_energies[lev][-1] * acc[-1][0][1]
            spec_e += access_energies[lev + 1][-1] * acc[-1][1][1]
            spec_e += access_energies[lev + 1][-1] * acc[-1][1][0]

            tot_acc_for_wbw += acc[-1][1][0]
            tot_acc_for_rbw += acc[-1][1][1]
            if bw[lev + 1]:
                spec_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    spec_lat,
                )

        if (spec_best_cost is None) or ((spec_e * spec_lat) < spec_best_cost):
            spec_best_cost = spec_e * spec_lat
            spec_best_cycle = spec_lat
            spec_best_e = spec_e
            spec_dr_e = spec_e
            spec_best_ord = list(order)
    if spec_best_cost and spec_best_cost < (energy * bl_lat) and spec:
        return spec_best_cost, spec_best_ord, spec_best_cycle, spec_best_e, spec_dr_e
    return energy * bl_lat, ordering, bl_lat, energy, dr_e


def get_cost_edp_bypass(
    prob, accesses_all, e_mat, bw, active_pes=1, spec=True, dump=False
):
    src.counter.counter += 1

    # TODO merge with get_cost (low priority)
    macs = 1
    for _, v in prob.bounds.items():
        macs *= v
    bl_lat = macs / active_pes
    best_lat = bl_lat

    e_mat, op_e, bypassed, op_bypassed = e_mat

    mac_e = macs * 0.07e-12

    accesses, spec_accesses = accesses_all

    op_accesses = prob.get_operand_accesses()

    op_access_e = 0

    for i, access in enumerate(op_accesses[:-1]):
        bypassed_sp_op = 1
        for j in range(0, op_bypassed[i]):
            bypassed_acc = accesses[j][list(accesses[j].keys())[0]][i]
            bypassed_sp_op *= bypassed_acc[0] / bypassed_acc[1]
        op_access_e += access * op_e[i] / bypassed_sp_op

    bypassed_sp_op = 1
    for j in range(0, op_bypassed[-1]):
        bypassed_acc = accesses[j][list(accesses[j].keys())[0]][-1]
        bypassed_sp_op *= bypassed_acc[0][0] / bypassed_acc[1][0]
    op_access_e += (op_accesses[-1][0] + op_accesses[-1][1]) * op_e[-1] / bypassed_sp_op

    ordering = []
    energy = mac_e + op_access_e
    for lev, access in enumerate(accesses):
        loc_best_cost = None
        loc_best_e = None
        loc_best_lat = None
        loc_best_ord = None

        ord_to_try = access.keys()

        for k in ord_to_try:
            v = access[k]
            loc_e = 0
            loc_lat = bl_lat

            tot_acc_for_rbw = 0  # for read bw
            tot_acc_for_wbw = 0  # for write bw
            for i, vv in enumerate(v[:-1]):
                loc_e += e_mat[lev][i][0] * vv[0]
                loc_e += e_mat[lev][i][1] * vv[1]
                tot_acc_for_rbw += vv[1]
            loc_e += e_mat[lev][-1][0] * v[-1][0][1]
            loc_e += e_mat[lev][-1][1] * v[-1][1][1]
            loc_e += e_mat[lev][-1][1] * v[-1][1][0]
            tot_acc_for_wbw += v[-1][1][0]
            tot_acc_for_rbw += v[-1][1][1]

            # update latency (detect throttling)
            if bw[lev + 1]:
                loc_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    loc_lat,
                )

            loc_cost = (energy + loc_e) * loc_lat

            if (loc_best_cost is None) or (loc_cost < loc_best_cost):
                loc_best_cost = loc_cost
                loc_best_e = loc_e
                loc_best_lat = loc_lat
                loc_best_ord = k

        ordering.append(loc_best_ord)
        energy += loc_best_e
        # update baseline/default latency
        bl_lat = max(bl_lat, loc_best_lat)

    spec_best_cost = None
    for order, access in spec_accesses.items():
        spec_e = mac_e + op_access_e
        spec_lat = best_lat

        for lev, acc in enumerate(access):
            tot_acc_for_rbw = 0
            tot_acc_for_wbw = 0
            for i, vv in enumerate(acc[:-1]):
                spec_e += e_mat[lev][i][0] * vv[0]
                spec_e += e_mat[lev][i][1] * vv[1]
                tot_acc_for_rbw += vv[1]
            spec_e += e_mat[lev][-1][0] * acc[-1][0][1]
            spec_e += e_mat[lev][-1][1] * acc[-1][1][1]
            spec_e += e_mat[lev][-1][1] * acc[-1][1][0]
            tot_acc_for_wbw += acc[-1][1][0]
            tot_acc_for_rbw += acc[-1][1][1]

            if bw[lev + 1]:
                spec_lat = max(
                    tot_acc_for_rbw / bw[lev + 1][0],
                    tot_acc_for_wbw / bw[lev + 1][1],
                    spec_lat,
                )

        if (spec_best_cost is None) or ((spec_e * spec_lat) < spec_best_cost):
            spec_best_cost = spec_e * spec_lat
            spec_best_cycle = spec_lat
            spec_best_e = spec_e
            spec_best_ord = list(order)

    if spec_best_cost and spec_best_cost < (energy * bl_lat) and spec:
        return spec_best_cost, spec_best_ord, spec_best_cycle, spec_best_e

    return energy * bl_lat, ordering, bl_lat, energy


def get_e_mat_with_bypass(access_energies, bypass):
    ll_i = len(access_energies) - 1
    mat_transpose = []
    op_e = []
    bypassed = []
    op_bypassed = []
    for tens in range(len(bypass[0])):
        mat_transpose.append([])
        bypassed.append([])
        ids = []
        # for each tensor, find levels that are not bypassed
        for lev in range(len(bypass)):
            if not bypass[lev][tens]:
                ids.append(lev)
                if len(ids) == 1:
                    op_e.append(access_energies[lev][tens])
                    op_bypassed.append(lev)
        # pad head of list with 0s
        if ids and ids[0] > 0:
            for i in range(ids[0]):
                mat_transpose[tens].append((0, 0))
        for i, lev in enumerate(ids):
            upper = ids[i + 1] if (i + 1) < len(ids) else ll_i
            mat_transpose[tens].append(
                (access_energies[lev][tens], access_energies[upper][tens])
            )
            bypassed[tens].append((lev, upper))
            # pad with zeros
            for j in range(upper - lev - 2):
                mat_transpose[tens].append((0, 0))
        # pad with zeros if every level is bypassed
        if not ids:
            for lev in range(len(bypass)):
                mat_transpose[tens].append((0, 0))
            op_e.append(access_energies[ll_i][tens])
            op_bypassed.append(ll_i)
            bypassed[tens].append(range(ll_i + 1))
    ret = [[] for _ in access_energies]
    for tens, _ in enumerate(mat_transpose):
        for lev, _ in enumerate(mat_transpose[tens]):
            ret[lev].append(mat_transpose[tens][lev])
    return ret, op_e, bypassed, op_bypassed


def get_sptl_utl(sp, ttl_pe):
    utlzd = 1
    for axs in sp[:-1]:
        for dim in axs:
            utlzd *= axs[dim]
    return utlzd / ttl_pe


def dvd_mpng_amng_bfrs(gnrc_tl):
    tls = gnrc_tl.subtiles
    pr_bfr_tl = []
    for lvl_i, tmprl_lvl in enumerate(tls):
        if gnrc_tl.spatial[lvl_i]:
            continue
        pr_bfr_tl.append([copy.deepcopy(tmprl_lvl) for _ in range(3)])
    return pr_bfr_tl


def genrc_tl_sz(gnrc_tl):
    tls = gnrc_tl.subtiles
    pr_bfr_tl = dvd_mpng_amng_bfrs(gnrc_tl)
    sizes = [[0, 0, 0] for _ in pr_bfr_tl]
    nn_sptl_idx = -1
    for lvl_i, tmprl_lvl in enumerate(tls):
        if gnrc_tl.spatial[lvl_i]:
            continue
        nn_sptl_idx += 1
        n = tls[lvl_i]["N"]
        c = tmprl_lvl["C"]
        k = tmprl_lvl["K"]
        r = tmprl_lvl["R"]
        s = tmprl_lvl["S"]
        p = tmprl_lvl["P"]
        q = tmprl_lvl["Q"]
        for lwr_lvl in tls[:lvl_i]:
            n *= lwr_lvl["N"]
            c *= lwr_lvl["C"]
            k *= lwr_lvl["K"]
            r *= lwr_lvl["R"]
            s *= lwr_lvl["S"]
            p *= lwr_lvl["P"]
            q *= lwr_lvl["Q"]
        sizes[nn_sptl_idx][0] = n * c * (p + r - 1) * (q + s - 1)
        sizes[nn_sptl_idx][1] = k * c * r * s
        sizes[nn_sptl_idx][2] = n * k * p * q
    return sizes


def fctrs(n):
    ret = sorted(
        list(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
    )
    return ret


def fx_unevn_L0(pr_bfr_tls, sprt_strct, memrs):
    tnsr_dims = [
        ["K"],
        ["N", "P", "Q"],
        ["C", "R", "S"],
    ]
    pr_bfr_unevn = copy.deepcopy(pr_bfr_tls)
    for i in pr_bfr_unevn:
        for j in i:
            for x in j:
                j[x] = 1

    lvl = 0
    for tnsr_i, tnsr_bfr in enumerate(sprt_strct[lvl]):
        if not tnsr_bfr:
            continue
        tl = pr_bfr_tls[lvl][tnsr_i]
        upr_tl = pr_bfr_tls[lvl + 1][tnsr_i]
        for dim in tnsr_dims[tnsr_i]:
            # get the factors
            ptnl_fctrs = fctrs(upr_tl[dim])
            for ptnl_fctr in reversed(list(ptnl_fctrs)):
                if ptnl_fctr == 1:
                    continue
                cp_pr_bfr_tls = copy.deepcopy(pr_bfr_tls)

                cp_pr_bfr_tls[lvl][tnsr_i][dim] = (
                    cp_pr_bfr_tls[lvl][tnsr_i][dim] * ptnl_fctr
                )
                for x in range(3):
                    cp_pr_bfr_tls[lvl + 1][x][dim] = int(
                        cp_pr_bfr_tls[lvl + 1][x][dim] / ptnl_fctr
                    )
                if tnsr_i == 0:
                    # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                    pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                    pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                    break
                elif tnsr_i == 1:
                    ifmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                            - 1
                        )
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                            - 1
                        )
                    )
                    if ifmap_tnsr_sz < memrs["L1_ifmap"] or not sprt_strct[lvl][0]:
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break
                elif tnsr_i == 2:
                    w_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["K"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                    )
                    ifmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * pr_bfr_unevn[lvl][tnsr_i]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * pr_bfr_unevn[lvl][tnsr_i]["C"]
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                            * pr_bfr_unevn[lvl][tnsr_i]["P"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                            * pr_bfr_unevn[lvl][tnsr_i]["R"]
                            - 1
                        )
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                            * pr_bfr_unevn[lvl][tnsr_i]["Q"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                            * pr_bfr_unevn[lvl][tnsr_i]["S"]
                            - 1
                        )
                    )
                    if (w_tnsr_sz < memrs["L1_w"] or not sprt_strct[lvl][1]) and (
                        ifmap_tnsr_sz < memrs["L1_ifmap"] or not sprt_strct[lvl][0]
                    ):
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break
    return pr_bfr_tls, pr_bfr_unevn


def fx_unevn_L1_sm_bfr(pr_bfr_tls, sprt_strct, memrs, pr_bfr_unevn, sbtls):
    tnsr_dims = [
        ["N", "C", "P", "Q", "R", "S"],
        ["K", "C", "R", "S"],
        ["N", "K", "P", "Q"],
    ]

    lvl = 1
    for tnsr_i, tnsr_bfr in enumerate(sprt_strct[lvl]):
        if not tnsr_bfr:
            continue
        tl = pr_bfr_tls[lvl][tnsr_i]
        upr_tl = pr_bfr_tls[lvl + 1][tnsr_i]
        for dim in tnsr_dims[tnsr_i]:
            # get the factors
            ptnl_fctrs = fctrs(upr_tl[dim])
            for ptnl_fctr in reversed(list(ptnl_fctrs)):
                if ptnl_fctr == 1:
                    continue
                cp_pr_bfr_tls = copy.deepcopy(pr_bfr_tls)

                cp_pr_bfr_tls[lvl][tnsr_i][dim] = (
                    cp_pr_bfr_tls[lvl][tnsr_i][dim] * ptnl_fctr
                )
                for x in range(3):
                    cp_pr_bfr_tls[lvl + 1][x][dim] = int(
                        cp_pr_bfr_tls[lvl + 1][x][dim] / ptnl_fctr
                    )
                if tnsr_i == 0:
                    ifmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * sbtls[1]["N"]
                        * cp_pr_bfr_tls[0][tnsr_i]["N"]
                        * pr_bfr_unevn[0][0]["N"]
                        * pr_bfr_unevn[0][1]["N"]
                        * pr_bfr_unevn[0][2]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * sbtls[1]["C"]
                        * cp_pr_bfr_tls[0][tnsr_i]["C"]
                        * pr_bfr_unevn[0][0]["C"]
                        * pr_bfr_unevn[0][1]["C"]
                        * pr_bfr_unevn[0][2]["C"]
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                            * sbtls[1]["P"]
                            * cp_pr_bfr_tls[0][tnsr_i]["P"]
                            * pr_bfr_unevn[0][0]["P"]
                            * pr_bfr_unevn[0][1]["P"]
                            * pr_bfr_unevn[0][2]["P"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                            * sbtls[1]["R"]
                            * cp_pr_bfr_tls[0][tnsr_i]["R"]
                            * pr_bfr_unevn[0][0]["R"]
                            * pr_bfr_unevn[0][1]["R"]
                            * pr_bfr_unevn[0][2]["R"]
                            - 1
                        )
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                            * sbtls[1]["Q"]
                            * cp_pr_bfr_tls[0][tnsr_i]["Q"]
                            * pr_bfr_unevn[0][0]["Q"]
                            * pr_bfr_unevn[0][1]["Q"]
                            * pr_bfr_unevn[0][2]["Q"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                            * sbtls[1]["S"]
                            * cp_pr_bfr_tls[0][tnsr_i]["S"]
                            * pr_bfr_unevn[0][0]["S"]
                            * pr_bfr_unevn[0][1]["S"]
                            * pr_bfr_unevn[0][2]["S"]
                            - 1
                        )
                    )
                    if ifmap_tnsr_sz < memrs["L1_ifmap"]:
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break

                elif tnsr_i == 1:
                    w_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * sbtls[1]["C"]
                        * cp_pr_bfr_tls[0][tnsr_i]["C"]
                        * pr_bfr_unevn[0][0]["C"]
                        * pr_bfr_unevn[0][1]["C"]
                        * pr_bfr_unevn[0][2]["C"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["K"]
                        * sbtls[1]["K"]
                        * cp_pr_bfr_tls[0][tnsr_i]["K"]
                        * pr_bfr_unevn[0][0]["K"]
                        * pr_bfr_unevn[0][1]["K"]
                        * pr_bfr_unevn[0][2]["K"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                        * sbtls[1]["R"]
                        * cp_pr_bfr_tls[0][tnsr_i]["R"]
                        * pr_bfr_unevn[0][0]["R"]
                        * pr_bfr_unevn[0][1]["R"]
                        * pr_bfr_unevn[0][2]["R"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                        * sbtls[1]["S"]
                        * cp_pr_bfr_tls[0][tnsr_i]["S"]
                        * pr_bfr_unevn[0][0]["S"]
                        * pr_bfr_unevn[0][1]["S"]
                        * pr_bfr_unevn[0][2]["S"]
                    )
                    if w_tnsr_sz < memrs["L1_w"]:
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break

                elif tnsr_i == 2:
                    ofmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * sbtls[1]["N"]
                        * cp_pr_bfr_tls[0][tnsr_i]["N"]
                        * pr_bfr_unevn[0][0]["N"]
                        * pr_bfr_unevn[0][1]["N"]
                        * pr_bfr_unevn[0][2]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["K"]
                        * sbtls[1]["K"]
                        * cp_pr_bfr_tls[0][tnsr_i]["K"]
                        * pr_bfr_unevn[0][0]["K"]
                        * pr_bfr_unevn[0][1]["K"]
                        * pr_bfr_unevn[0][2]["K"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                        * sbtls[1]["P"]
                        * cp_pr_bfr_tls[0][tnsr_i]["P"]
                        * pr_bfr_unevn[0][0]["P"]
                        * pr_bfr_unevn[0][1]["P"]
                        * pr_bfr_unevn[0][2]["P"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                        * sbtls[1]["Q"]
                        * cp_pr_bfr_tls[0][tnsr_i]["Q"]
                        * pr_bfr_unevn[0][0]["Q"]
                        * pr_bfr_unevn[0][1]["Q"]
                        * pr_bfr_unevn[0][2]["Q"]
                    )
                    if ofmap_tnsr_sz < memrs["L1_ofmap"]:
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break
    return pr_bfr_tls, pr_bfr_unevn


def fx_unevn_L1(pr_bfr_tls, sprt_strct, memrs, pr_bfr_unevn, sbtls):
    tnsr_dims = [
        ["K"],
        ["N", "P", "Q"],
        ["C", "R", "S"],
    ]

    lvl = 1
    for tnsr_i, tnsr_bfr in enumerate(sprt_strct[lvl]):
        if not tnsr_bfr:
            continue
        tl = pr_bfr_tls[lvl][tnsr_i]
        upr_tl = pr_bfr_tls[lvl + 1][tnsr_i]
        for dim in tnsr_dims[tnsr_i]:
            # get the factors
            ptnl_fctrs = fctrs(upr_tl[dim])
            for ptnl_fctr in reversed(list(ptnl_fctrs)):
                if ptnl_fctr == 1:
                    continue
                cp_pr_bfr_tls = copy.deepcopy(pr_bfr_tls)

                cp_pr_bfr_tls[lvl][tnsr_i][dim] = (
                    cp_pr_bfr_tls[lvl][tnsr_i][dim] * ptnl_fctr
                )
                for x in range(3):
                    cp_pr_bfr_tls[lvl + 1][x][dim] = int(
                        cp_pr_bfr_tls[lvl + 1][x][dim] / ptnl_fctr
                    )
                if tnsr_i == 0:
                    # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                    pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                    pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                    break
                elif tnsr_i == 1:
                    ifmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * sbtls[1]["N"]
                        * cp_pr_bfr_tls[0][tnsr_i]["N"]
                        * pr_bfr_unevn[0][tnsr_i]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * sbtls[1]["C"]
                        * cp_pr_bfr_tls[0][tnsr_i]["C"]
                        * pr_bfr_unevn[0][tnsr_i]["C"]
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                            * sbtls[1]["P"]
                            * cp_pr_bfr_tls[0][tnsr_i]["P"]
                            * pr_bfr_unevn[0][tnsr_i]["P"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                            * sbtls[1]["R"]
                            * cp_pr_bfr_tls[0][tnsr_i]["R"]
                            * pr_bfr_unevn[0][tnsr_i]["R"]
                            - 1
                        )
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                            * sbtls[1]["Q"]
                            * cp_pr_bfr_tls[0][tnsr_i]["Q"]
                            * pr_bfr_unevn[0][tnsr_i]["Q"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                            * sbtls[1]["S"]
                            * cp_pr_bfr_tls[0][tnsr_i]["S"]
                            * pr_bfr_unevn[0][tnsr_i]["S"]
                            - 1
                        )
                    )
                    if ifmap_tnsr_sz < memrs["L1_ifmap"] or not sprt_strct[lvl][0]:
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break
                elif tnsr_i == 2:
                    w_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * sbtls[1]["C"]
                        * cp_pr_bfr_tls[0][tnsr_i]["C"]
                        * pr_bfr_unevn[0][tnsr_i]["C"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["K"]
                        * sbtls[1]["K"]
                        * cp_pr_bfr_tls[0][tnsr_i]["K"]
                        * pr_bfr_unevn[0][tnsr_i]["K"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                        * sbtls[1]["R"]
                        * cp_pr_bfr_tls[0][tnsr_i]["R"]
                        * pr_bfr_unevn[0][tnsr_i]["R"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                        * sbtls[1]["S"]
                        * cp_pr_bfr_tls[0][tnsr_i]["S"]
                        * pr_bfr_unevn[0][tnsr_i]["S"]
                    )
                    ifmap_tnsr_sz = (
                        cp_pr_bfr_tls[lvl][tnsr_i]["N"]
                        * sbtls[1]["N"]
                        * cp_pr_bfr_tls[0][tnsr_i]["N"]
                        * pr_bfr_unevn[0][tnsr_i]["N"]
                        * pr_bfr_unevn[lvl][tnsr_i]["N"]
                        * cp_pr_bfr_tls[lvl][tnsr_i]["C"]
                        * sbtls[1]["C"]
                        * cp_pr_bfr_tls[0][tnsr_i]["C"]
                        * pr_bfr_unevn[0][tnsr_i]["C"]
                        * pr_bfr_unevn[lvl][tnsr_i]["C"]
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["P"]
                            * sbtls[1]["P"]
                            * cp_pr_bfr_tls[0][tnsr_i]["P"]
                            * pr_bfr_unevn[0][tnsr_i]["P"]
                            * pr_bfr_unevn[lvl][tnsr_i]["P"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["R"]
                            * sbtls[1]["R"]
                            * cp_pr_bfr_tls[0][tnsr_i]["R"]
                            * pr_bfr_unevn[0][tnsr_i]["R"]
                            * pr_bfr_unevn[lvl][tnsr_i]["R"]
                            - 1
                        )
                        * (
                            cp_pr_bfr_tls[lvl][tnsr_i]["Q"]
                            * sbtls[1]["Q"]
                            * cp_pr_bfr_tls[0][tnsr_i]["Q"]
                            * pr_bfr_unevn[0][tnsr_i]["Q"]
                            * pr_bfr_unevn[lvl][tnsr_i]["Q"]
                            + cp_pr_bfr_tls[lvl][tnsr_i]["S"]
                            * sbtls[1]["S"]
                            * cp_pr_bfr_tls[0][tnsr_i]["S"]
                            * pr_bfr_unevn[0][tnsr_i]["S"]
                            * pr_bfr_unevn[lvl][tnsr_i]["S"]
                            - 1
                        )
                    )
                    if (w_tnsr_sz < memrs["L1_w"] or not sprt_strct[lvl][1]) and (
                        ifmap_tnsr_sz < memrs["L1_ifmap"] or not sprt_strct[lvl][0]
                    ):
                        # pr_bfr_tls[lvl][tnsr_i] = cp_pr_bfr_tls[lvl][tnsr_i]
                        pr_bfr_unevn[lvl][tnsr_i][dim] = ptnl_fctr
                        pr_bfr_tls[lvl + 1] = copy.deepcopy(cp_pr_bfr_tls[lvl + 1])
                        break
    return pr_bfr_tls, pr_bfr_unevn
