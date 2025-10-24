import sys, os
sys.path.append(os.getcwd() + "/src")
import time, copy, os
from math import ceil, gcd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce
from tile_graph import GenericTileGraph
from tiling import *
from optimization_utils import *

# ****************************** prune spatial+temporal level ******************************
"""
    For each unrolling in the list of unrollings passed, creates a tree for the problem (full layer divided by the unrolled bounds) 
    and order. It then returns tilings that make it through the tree 
    returns: 
        1- pairs of a tile (that made it through the tree) and an order as a list 
"""

def thread_task(my_args):
    spatial_candids = my_args[0]
    small_prob_glb = my_args[1]
    prob = my_args[2]
    mem = my_args[3]
    order = my_args[4]
    smallest_prob = my_args[5]
    tile = my_args[6]
    static = my_args[7]
    prior = my_args[8]
    bw = my_args[9]
    access_energies = my_args[10]
    edp = my_args[11]
    spec = my_args[12]
    next_lev = my_args[13]
    bypass = my_args[14]
    tens_desc = prob.tens_desc

    ret = []

    # ******* go through the spatial candidates *******
    for sp in spatial_candids:
        # ******* divide the total problem by the unrolled dimensions 
        #        to get the subproblem                               *******
        if len(tile.subtiles) == 1:
            small_prob = tile.subtiles[-1].copy()
            for k, v in small_prob.items():
                small_prob[k] = int(v / sp[0][k] / sp[1][k])

            # ******* create a tile graph (search tree to find the good tilings) *******
            inner_tile_graph = GenericTileGraph(
                small_prob,
                smallest_prob,
                prob.tens_desc,
            )
            
            # ******* based on the order, some dimensions will be static meaning 
            #        the tree will not grow in those dimensions (these are the non-indexing 
            #         dimensions of the reused operand inside the tile)                     *******
            if static:
                static_list = []
                for pot_ord in prob.tens_to_ord[sp[2]]:
                    static_list.append(prob.order_to_static[pot_ord])
                if next_lev:
                    pot_tiles = inner_tile_graph.bottom_up_static_next(
                        mem, next_lev, sp, static=static_list
                    )
                else:
                    pot_tiles = inner_tile_graph.bottom_up_static(
                        mem, static=static_list
                    )
            else:
                if next_lev:
                    pot_tiles = inner_tile_graph.bottom_up_next(
                        mem, next_lev, sp, bypass=bypass
                    )
                else:
                    pot_tiles = inner_tile_graph.bottom_up(mem, bypass=bypass)

            # ******* for all the leaves of the tile graph, create a tiling configuration 
            #         that consists of the original unrolling we were searching for, the tile, 
            #         and the remaining upper factors                                           *******
            for pot_tile in pot_tiles:
                temp_tile = pot_tile.copy()
                for k, v in pot_tile.items():
                    temp_tile[k] *= sp[0][k] * sp[1][k]
                final_tile = (
                    tile.split(temp_tile, tail=False)
                    .split(sp[0], spatial=True, tail=False, pred=False)
                    .split(sp[1], spatial=True, tail=False, pred=False)
                )
                ret.append((final_tile, order))
    
        else:
            small_prob = copy.deepcopy(smallest_prob)
            for st in tile.subtiles[:-1]:
                for k, v in st.items():
                    small_prob[k] *= v
            split_tile = tile.subtiles[-1].copy()
            for k, v in split_tile.items():
                split_tile[k] = int(v / sp[0][k] / sp[1][k])
            tile_graph = GenericTileGraph(split_tile, small_prob, prob.tens_desc)
            if static:
                static_list = []
                for pot_ord in prob.tens_to_ord[sp[2]]:
                    static_list.append(prob.order_to_static[pot_ord])
                pot_tiles = tile_graph.bottom_up_static(mem, static=static_list)
            elif prior:
                pot_tiles = tile_graph.bottom_up(mem, prior=order)
            else:
                pot_tiles = tile_graph.bottom_up(mem, bypass=bypass[-1])
            for pot_tile in pot_tiles:
                final_tile = (
                    tile.split(pot_tile, tail=True)
                    .split(sp[0], spatial=True, tail=True)
                    .split(sp[1], spatial=True, tail=True)
                )
                ret.append((final_tile, order))
    return ret



# ****************************** prune spatial+temporal level ******************************
"""
get unrolling candidates by calling enumerate_spatial(). Then divides it between threads
and calls them to prune the space and find best tiles candids.

returns:
    pairs of a tile (that made it through the tree) and an order as a list -> the list is aggregated
    result of all threads
"""

def tls_sptl_mlt_thrd(
    tiles,
    prob,
    mem,
    x_axis,
    y_axis,
    static=False,
    prior=False,
    num_threads=8,
    bw=None,
    access_energies=None,
    edp=True,
    spec=True,
    bypass=None,
    next_lev=None,
    sptl_cnstrnts=[{}, {}],
    sptl_utlzn_cnstrnts=1.0
):

    # ****** enumerate dimensions that should be unrolled together 
    #       based on Sunstone unrolling principle                 ******
    ret = []
    for tile, order in tiles:
        unroll_to_try = []
        for i in range(len(prob.tens_desc)):
            for unroll in prob.tens_to_sp2[i]:
                unroll_to_try.append((unroll, i))

        # ****** get potential unrollings based on Sunstone principle & 
        #        layer configuration                                    ******
        
        unrln_cndds_len = 0
        lp_itr = 0
        while(unrln_cndds_len == 0):
            lp_itr+=1
            assert (lp_itr < 4), "Sunstone couldn't find any unrolling. Try changing the spatial constraint provided or increasing the size of your buffers."
            if (lp_itr == 1):
                zx = enumerate_spatial(prob, tile.subtiles[-1], unroll_to_try, x_axis, y_axis)
            elif (lp_itr == 2):
                zx = enumerate_spatial(prob, tile.subtiles[-1], unroll_to_try, x_axis, y_axis, nd=False)
            elif (lp_itr == 3):
                zx = enumerate_spatial(prob, tile.subtiles[-1], unroll_to_try, x_axis, y_axis, nd=False, max_util=False)
            #for x in zx:
            #    print(x)
            if sptl_cnstrnts:
                zx_fltrd = []
                for unrl in zx:
                    if  ((not sptl_cnstrnts[0]) or (all(sptl_cnstrnts[0][dim] == unrl[0][dim] for dim in sptl_cnstrnts[0]))) and\
                        ((not sptl_cnstrnts[1]) or (all(sptl_cnstrnts[1][dim] == unrl[1][dim] for dim in sptl_cnstrnts[1]))):
                        zx_fltrd.append(unrl)
                for unrl in zx:
                    if  ((not sptl_cnstrnts[0]) or (all(sptl_cnstrnts[0][dim] == unrl[1][dim] for dim in sptl_cnstrnts[0]))) and\
                        ((not sptl_cnstrnts[1]) or (all(sptl_cnstrnts[1][dim] == unrl[0][dim] for dim in sptl_cnstrnts[1]))) and\
                        unrl not in zx_fltrd:
                            zx_fltrd.append(unrl)
                #assert len(zx_fltrd) > 0, "There was no unrolling that matches the spatial constraints provided"
                zx = zx_fltrd
            unrln_cndds_len = len(zx)

        # ******* Now we want to find the potential tilings for each 
        #         unrolling candidate. To do so, we divide the total 
        #         unrollings we have between our working threads & then     
        #         create a tile graph (explained in the paper) for each     
        #         unrolling from within the threads                     *******
        num_unrollings = len(zx)
        unrollings_per_thread = int(num_unrollings / num_threads)
        left_at_end = num_unrollings % num_threads
        work_distributed = []

        for thread_id in range(num_threads):
            # List of Thread arguments:
            #   1- spatial_candids
            #   2- small_prob
            #   3- prob
            #   4- mem
            #   5- order
            #   6- smallest_prob
            #   7- tile
            #   8- static
            #   9- prior
            #   10- bw
            #   11- access_energies
            #   12- edp
            #   13- spec
            #   14- next_level
            #   15- by_pass

            small_prob = tile.subtiles[-1]
            prob_dup = prob
            local_smallest_prob = smallest_prob(prob)
            local_tile = tile
            local_order = order
            local_mem = mem

            # Distribute work to threads
            begin_i = thread_id * unrollings_per_thread
            end_i = (thread_id + 1) * unrollings_per_thread
            if thread_id == num_threads - 1:
                end_i += left_at_end
            work_distributed.append(
                [
                    zx[begin_i:end_i],
                    small_prob,
                    prob_dup,
                    local_mem,
                    local_order,
                    local_smallest_prob,
                    local_tile,
                    static,
                    prior,
                    bw,
                    access_energies,
                    edp,
                    spec,
                    next_lev,
                    bypass,
                ]
            )
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for process_res in executor.map(thread_task, work_distributed):
                ret += process_res
    return ret



# ****************************** Spatio-Temporal tiling of a memory level ******************************
"""
    get candidates by calling get_sptl_tls_prll() and calculate the estimated cost for each
    returns:
        1- pairs of a tile and list of best order at each level for that tile as a list
        2- estimated cost of each tile-order pair as a list
"""
def tls_sptl_tmprl(
    tiles,
    prob,
    mem_size,
    access_energies,
    x_axis=None,
    y_axis=None,
    static=False,
    prior=False,
    edp=False,
    bw=None,
    threads=8,
    spec_cost=True,
    bypass=None,
    next_lev=None,
    sptl_cnstrnts=[[], []],
):
    # ****** get potential unrolling & potential tilings for each unrolling ******
    tl_cndds = tls_sptl_mlt_thrd(
        tiles,
        prob,
        mem_size,
        x_axis,
        y_axis,
        static=static,
        prior=prior,
        num_threads=threads,
        bw=bw,
        access_energies=access_energies,
        edp=edp,
        spec=spec_cost,
        next_lev=next_lev,
        bypass=bypass,
        sptl_cnstrnts=sptl_cnstrnts,
    )

    # ****** adjust accesses & energy based on by-passing 
    #        (no access is made for the by-passed operands of a level) ******
    if bypass:
        e_mat = get_e_mat_with_bypass(access_energies, bypass)

    tile_ordr_pairs = []
    costs = []
    for pot_tl, _ in tl_cndds:

        if not bw:
            bw = [None] * len(pot_tl.subtiles)

        if bypass:
            cost, order, latency, energy = get_cost_edp_bypass(
                prob,
                pot_tl.get_accesses(),
                e_mat,
                bw=bw,
                active_pes=pot_tl.get_spatial_count(),
                spec=False,
            )
        else:
            cost, order, latency, energy = get_cost_edp(
                prob,
                pot_tl.get_accesses(),
                access_energies,
                bw,
                active_pes=pot_tl.get_spatial_count(),
                spec=False,
            )
        cost = (cost, latency, energy)
        tile_ordr_pairs.append((pot_tl, order))
        costs.append(cost)

    return tile_ordr_pairs, costs


"""
******* 
    This is the twin function of tls_sptl_tmprl, but also calculate the cost  
    of the tiling levels available so far for alpha-beta pruning
*******
"""
def tls_sptl_tmprl_alpha_beta(
    tiles,
    prob,
    mem_size,
    access_energies,
    costs,
    x_axis=None,
    y_axis=None,
    static=False,
    prior=False,
    bw=None,
    bypass=None,
    threads=8
):
    best_cost = None
    best_tiles = []

    if bypass:
        e_mat = get_e_mat_with_bypass(access_energies, bypass)

    macs = 1
    for _, v in prob.bounds.items():
        macs *= v

    sorted_costs = sorted(enumerate(costs), key=lambda x: x[1])
    e_opt = sum(
        [
            x * (access_energies[-2][0] + access_energies[-1][0])
            for x in prob.get_prob_size()
        ]
    )
    e_opt = 0
    costs = []
    sp_opt = (x_axis if x_axis else 1) * (y_axis if y_axis else 1)
    for i, partial_cost in sorted_costs:
        opt = e_opt * (macs / (tiles[i][0].get_spatial_count() * sp_opt))
        if best_cost is not None and (partial_cost[0] / sp_opt + opt) > best_cost:
            break

        L2_ret = tls_sptl_tmprl(
            tiles=[(tiles[i])],
            prob=prob,
            mem_size=mem_size,
            access_energies=access_energies,
            x_axis=x_axis,
            y_axis=y_axis,
            static=static,
            bw=bw,
            bypass=bypass,
            threads=threads
        )
        L2_cndds = [x[0] for x in L2_ret[0]]
        for new_pot_tile in L2_cndds:

            if not bw:
                bw = [None] * len(new_pot_tile.subtiles)
            
            if bypass:
                cost, order, latency, energy = get_cost_edp_bypass(
                    prob,
                    new_pot_tile.get_accesses(),
                    e_mat,
                    bw=bw,
                    active_pes=new_pot_tile.get_spatial_count(),
                    spec=False,
                )
            else:
                cost, order, latency, energy = get_cost_edp(
                    prob,
                    new_pot_tile.get_accesses(),
                    access_energies,
                    bw,
                    active_pes=new_pot_tile.get_spatial_count(),
                    spec=False,
                )

            cost = (cost, latency, energy)
            if best_cost is None or cost[0] < best_cost:
                best_cost = cost[0]
            best_tiles.append((new_pot_tile, order))
            costs.append(cost)
    return best_tiles, costs



# ****************************** prune spatial+temporal level ******************************
# - Creates a tree for the problem and order and returns tilings that make it through the tree
#   returns:
#     1- pairs of a tile (that made it through the tree) and an order as a list
def alpha_beta_thrd_task(args):
    mem_size = args[0]
    access_energies = args[1]
    costs = args[2]
    static = args[3]
    prior = args[4]
    bw = args[5]
    e_opt = args[6]
    macs = args[7]
    thread_id = args[8]
    stride = args[9]
    tile_orders = args[10]
    prob = args[11]
    bypass = args[12]
    x_axis = args[13]
    y_axis = args[14]
    e_mat = args[15]

    best_tiles = []
    best_cost = None
    tls_evltd = 0
    # TODO j -> i
    for j in range(thread_id, len(tile_orders), stride):
        i = costs[j][0]
        partial_cost = costs[j][1]
        opt = e_opt * (macs / tile_orders[i][0].get_spatial_count())

        if best_cost is not None and (partial_cost + opt) > best_cost:
            break

        nxt_lvl_tl_cndds = enumerate_tiles(
            [tile_orders[i]], 
            prob, 
            mem_size, 
            static=static, 
            prior=prior,
            bypass=bypass[-1] if bypass else None,
        )
        
        for new_pot_tile, _ in nxt_lvl_tl_cndds:

            if bypass:
                cost, order, latency, energy = get_cost_edp_bypass(
                    prob,
                    new_pot_tile.get_accesses(),
                    e_mat,
                    bw,
                    active_pes=new_pot_tile.get_spatial_count(),
                )
            else:
                cost, order, latency, energy = get_cost_edp(
                    prob,
                    new_pot_tile.get_accesses(),
                    access_energies,
                    bw,
                    active_pes=new_pot_tile.get_spatial_count(),
                )
            cost = (cost, latency, energy)
            if best_cost is None or cost[0] < best_cost:
                best_tiles = [(new_pot_tile, order, cost)]
                best_cost = cost[0]
            elif cost[0] == best_cost:
                best_tiles.append((new_pot_tile, order, cost))

    return best_tiles, best_cost

# ****************************** prune spatial+temporal level ******************************
# - Sorts the costs. Creates an upper bound for the cost. Distributes candidates over threads
#   and optimizes L2
#   returns:
#     1- best tile (list of L1, L2, DR tiles)
#     2- best order (list of L1, L2, DR orders)
#     3- best cost
def tls_tmprl_alpha_beta_mlt_thrd(
    tiles,
    prob,
    mem_size,
    access_energies,
    costs,
    x_axis=None,
    y_axis=None,
    static=False,
    prior=False,
    bw=None,
    threads=8,
    bypass=None,
):
    e_mat = None
    if bypass:
        e_mat = get_e_mat_with_bypass(access_energies, bypass)
    # creates upper bound for the cost
    macs = 1
    for _, v in prob.bounds.items():
        macs *= v

    sorted_costs = sorted(enumerate(costs), key=lambda x: x[1])

    # this is hardcoded and assumes this method is used for optimizing the level before DRAM
    # should be changed later
    e_opt = sum(
            [
                x * (access_energies[-2][0] + access_energies[-1][0])
                for x in prob.get_prob_size()
            ]
        )

    work_distributed = []
    for thread_id in range(threads):
        work_distributed.append(
            (
                mem_size,
                access_energies,
                sorted_costs,
                static,
                prior,
                bw,
                e_opt,
                macs,
                thread_id,
                threads,
                tiles,
                prob,
                bypass,
                x_axis,
                y_axis,
                e_mat,
            )
        )
    best_candids = []
    thrd_fnc = alpha_beta_thrd_task #if not smba else alpha_beta_smba_thrd_task
    with ProcessPoolExecutor(max_workers=threads) as executor:
        for ret in executor.map(thrd_fnc, work_distributed):
            best_candids.extend(ret[0])

        # find the candid with min cost among the best candids
        bst_cst = None
        bst_tl = None
        bst_ordr = None
        for candid in best_candids:
            tl = candid[0]
            ordr = candid[1]
            cst = candid[2][0]
            if bst_cst is None or cst < bst_cst:
                bst_tl = tl
                bst_ordr = ordr
                bst_cst = cst

        return bst_tl, bst_ordr, bst_cst
