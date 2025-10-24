from collections import deque
import math
import scipy.special as sp
from functools import reduce


class GenericTileGraph:
    # prob->dict (eg:  {"P" : p, "Q" : q etc (eg:  {"P" : p, "Q" : q etc})
    # small_prob->dict (same struct as prob)
    # tens_desc -> list [(),(),()]
    # [(("P", "R"), ("Q", "S"), "C", "N"), ("R", "S", "C", "K"), ("P", "Q", "K","N")]
    def __init__(self, prob, small_prob, tens_desc, banked=False, size_fun=None):
        self.prob = prob
        self.small_prob = small_prob
        self.tens_desc = tens_desc
        self.banked = banked
        self.size_fun = size_fun

    def get_size(self, node):
        if self.size_fun:
            return self.size_fun(node, self.small_prob)
        else:
            sizes = []
            # [[("P", "R"), ("Q", "S"), "N", "C"], ]
            # tens = [("P", "R"), ("Q", "S"), "N", "C"]
            # dim = ("P", "R") or ("Q", "S") or "N", "C"
            # dd = "P" or "R" or "2P"

            for tens in self.tens_desc:
                size = 1
                for dim in tens:

                    if type(dim) is tuple:  
                        d = 0
                        for dd in dim:
                            if len(dd) > 1:
                                d += int(dd[0]) * (
                                    node[dd[-1]] * self.small_prob[dd[-1]] - 1
                                )

                            else: # sliding window effect
                                d += node[dd] * self.small_prob[dd] - 1
                        d += 1
                        size *= d

                    else:   
                        if len(dim) > 1:   
                            size *= (
                                int(dim[0])
                                * (node[dim[-1]] * self.small_prob[dd[-1]] - 1)
                                + 1
                            )

                        else: # normal dims
                            size *= node[dim] * self.small_prob[dim]
                # size = 1
                # for dim in tens:
                # if len(dim) > 1:
                #    d  = 1
                #    for dd in dim:
                #        if len(sizes) == 0 and (dd == "P" or dd == "Q"): #len(dd) > 1:
                #            d += (2)*(node[dd[-1]]*self.small_prob[dd[-1]] - 1)
                #        else:
                #            d += (node[dd]*self.small_prob[dd] - 1)
                #    size *= (d + 1)
                # else:
                #    size *= (node[dim]*self.small_prob[dim])

                # d = sum([node[x]*self.small_prob[x] for x in dim]) - (len(dim) > 1)
                # size *= d
                sizes.append(size)
        return sizes

    def generate_db(self, mem_sizes):
        # TODO maybe have a better heuristic here, or factor this out as a param to the alg
        # essentially, if the memory sizes get too big (eg for L2), buidling and storing a >500k sparse array for O(1) access may be less efficent than just dynamically checking for the smallest mem, especially if the selection of memories is ~10
        if mem_sizes[-1] > 100000:
            return mem_sizes, "dynamic"
        db = [0]
        index = 0

        for i in range(1, mem_sizes[-1] + 1):
            if i > mem_sizes[index]:
                index += 1
            db.append(mem_sizes[index])

        return db, "static"

    def get_mem_size(self, node, db, db_type="static", split=False):
        sizes = self.get_size(node)

        if db_type == "static":
            if split:
                return tuple([db[x] if x <= len(db) else False for x in sizes])
            else:
                return db[sum(sizes)] if sum(sizes) < len(db) else False
        else:
            if split:
                mem_sizes = []
                for s in sizes:
                    found = False
                    for d in db:
                        if s <= d:
                            found = True
                            mem_sizes.append(s)
                            break
                    if not found:
                        mem_sizes.append(False)
                return tuple(mem_sizes)

            else:
                # TODO: may consider doing binary search, though this type is only considered if mem list is small
                for d in db:
                    if sum(sizes) <= d:
                        return d
                return False

    def valid_mem_config_area(self, mems, areas, thres):
        # if any of the tiles do not fit in their dedicated buffer, it would
        # be false instead of an integer so those can already rule out some configs
        for m in mems:
            if not m:
                return False
        return sum([areas[x] for x in mems]) <= thres

    def valid_mem_config_depth(self, mems, thres):
        return mems <= thres

    def dict_prod(self, tile):
        ret = 1
        for t in tile.values():
            ret *= t
        return ret

    def bottom_up_spatial(
        self, exact, traversed, max_util=True, nd=True, prior=False, min_occ=0
    ):
        # only check one bound
        if not nd:
            ret = []
            for b in traversed:
                if self.prob[b] % exact == 0:
                    sp = dict([(k, 1) for k in self.prob.keys()])
                    sp[b] = exact
                    ret.append(sp)
            return ret

        factors = {}

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        nodes_to_visit = deque([dict([(x, 1) for x in self.prob.keys()])])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()

            tot_bounds = 1
            for _, v in node.items():
                tot_bounds *= v

            if tot_bounds == exact:
                best_nodes.append(node)
            elif tot_bounds < exact:
                overflow = True
                for k, c in self.enumerate_children(node, factors).items():
                    if k in traversed:
                        tot_bounds = 1
                        for _, v in c.items():
                            tot_bounds *= v

                        # keep traversing the children if they fit
                        if tot_bounds <= exact:
                            overflow = False
                            if tuple(c.items()) not in visited:
                                visited.add(tuple(c.items()))
                                if (not prior) or self.pass_prior(c, factors, prior):
                                    nodes_to_visit.append(c)
                if (not max_util) and overflow and self.dict_prod(node) > min_occ:
                    best_nodes.append(node)

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        # return [self.convert_node_to_pair(n) for n in best_nodes]
        return best_nodes

    def bottom_up_static_mem(self, mems, split, thres, start=None, static=[], areas=[]):
        assert len(static)

        factors = {}

        graphs = []

        for sta in static:
            graph = "".join(self.prob.keys())
            for s in sta:
                graph = graph.replace(s, "")
            graphs.append(graph)

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # TODO update to sta/dyn (see below)
        db, db_type = self.generate_db(mems)

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([(dict([(x, 1) for x in self.prob.keys()]), graphs)])
        else:
            nodes_to_visit = deque([(start, graphs)])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        while nodes_to_visit:
            node, graphs = nodes_to_visit.popleft()

            children = self.enumerate_children(node, factors)
            children_to_add = dict(
                [(x, [False] * len(graphs)) for x in children.keys()]
            )
            add = False
            curr_mem = self.get_mem_size(node, db, db_type=db_type, split=True)
            for i, graph in enumerate(graphs):
                overflow_all = True  # TODO remove like below
                overflow_curr = True

                for edge in graph:
                    if edge in children:
                        child = children[edge]
                        child_mem = self.get_mem_size(child, db, split=True)
                        valid_mem = (
                            self.valid_mem_config_area(child_mem, areas, thres)
                            if split
                            else self.valid_mem_config_depth(child_mem, thres)
                        )
                        if valid_mem:
                            overflow_all = False

                            if child_mem == curr_mem:
                                overflow_curr = False

                            children_to_add[edge][i] = True

                # if none of the children fit, add the current node to the accepted list
                if overflow_all or overflow_curr:
                    add = True
            if add:
                best_nodes.append(node)

            for k, v in children_to_add.items():
                child_node = children[k]
                if tuple(child_node.items()) not in visited:
                    visited.add(tuple(child_node.items()))
                    next_graphs = [graphs[i] for i, x in enumerate(v) if x]
                    if next_graphs:
                        nodes_to_visit.append((child_node, next_graphs))

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        return best_nodes

    def bottom_up_static(
        self, mem, start=None, static=[], prior=None, banks=None, bypass=False
    ):
        assert len(static), static

        factors = {}

        graphs = []

        for sta in static:
            graph = "".join(self.prob.keys())
            for s in sta:
                graph = graph.replace(s, "")
            graphs.append(graph)

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([(dict([(x, 1) for x in self.prob.keys()]), graphs)])
        else:
            nodes_to_visit = deque([(start, graphs)])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        while nodes_to_visit:
            node, graphs = nodes_to_visit.popleft()
            # node)
            for xxx in factors:
                if len(factors[xxx]) == 0:
                    factors[xxx].append(1)
            # print(factors)

            children = self.enumerate_children(node, factors)
            children_to_add = dict(
                [(x, [False] * len(graphs)) for x in children.keys()]
            )
            add = False
            for i, graph in enumerate(graphs):
                overflow = True

                for edge in graph:
                    if edge in children:
                        child = children[edge]
                        if self.node_fits(child, mem, banks=None, bypass=bypass):
                            overflow = False
                            children_to_add[edge][i] = True

                if overflow:
                    for e, c in children.items():
                        if self.node_fits(c, mem, banks=None, bypass=bypass):
                            overflow = False
                            children_to_add[e][i] = True

                # if none of the children fit, add the current node to the accepted list
                if overflow:
                    add = True
            if add:
                best_nodes.append(node)

            for k, v in children_to_add.items():
                child_node = children[k]
                if tuple(child_node.items()) not in visited:
                    visited.add(tuple(child_node.items()))
                    next_graphs = [graphs[i] for i, x in enumerate(v) if x]
                    if next_graphs:
                        if (not prior) or (self.pass_prior(c, factors, prior)):
                            nodes_to_visit.append((child_node, next_graphs))

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        return best_nodes

    def bottom_up_static_next(
        self,
        mem,
        next_lev,
        spatial,
        start=None,
        static=[],
        prior=None,
        banks=None,
        bypass=False,
    ):
        assert len(static), static

        factors = {}

        graphs = []

        for sta in static:
            graph = "".join(self.prob.keys())
            for s in sta:
                graph = graph.replace(s, "")
            graphs.append(graph)

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([(dict([(x, 1) for x in self.prob.keys()]), graphs)])
        else:
            nodes_to_visit = deque([(start, graphs)])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        # TODO generalize
        inp_sp = (
            spatial[0]["P"]
            * spatial[1]["P"]
            * spatial[0]["Q"]
            * spatial[1]["Q"]
            * spatial[0]["C"]
            * spatial[1]["C"]
            * spatial[0]["N"]
            * spatial[1]["N"]
            * spatial[0]["R"]
            * spatial[1]["R"]
            * spatial[0]["S"]
            * spatial[1]["S"]
        )

        out_sp = (
            spatial[0]["P"]
            * spatial[1]["P"]
            * spatial[0]["Q"]
            * spatial[1]["Q"]
            * spatial[0]["K"]
            * spatial[1]["K"]
            * spatial[0]["N"]
            * spatial[1]["N"]
        )

        w_sp = (
            spatial[0]["R"]
            * spatial[1]["R"]
            * spatial[0]["S"]
            * spatial[1]["S"]
            * spatial[0]["C"]
            * spatial[1]["C"]
            * spatial[0]["K"]
            * spatial[1]["K"]
        )

        while nodes_to_visit:
            node, graphs = nodes_to_visit.popleft()
            # node)
            for xxx in factors:
                if len(factors[xxx]) == 0:
                    factors[xxx].append(1)
            # print(factors)

            children = self.enumerate_children(node, factors)
            children_to_add = dict(
                [(x, [False] * len(graphs)) for x in children.keys()]
            )
            add = False

            assert self.node_fits_next(
                node, next_lev, [inp_sp, w_sp, out_sp], banks=banks
            ) and self.node_fits(node, mem, banks=banks)

            for i, graph in enumerate(graphs):
                overflow = True

                for edge in graph:
                    if edge in children:
                        child = children[edge]

                        if self.node_fits(
                            child, mem, banks=banks, bypass=bypass
                        ) and self.node_fits_next(
                            child, next_lev, [inp_sp, w_sp, out_sp], banks=banks
                        ):
                            overflow = False
                            children_to_add[edge][i] = True

                if overflow:
                    for e, c in children.items():
                        if self.node_fits(
                            c, mem, banks=banks, bypass=bypass
                        ) and self.node_fits_next(
                            c, next_lev, [inp_sp, w_sp, out_sp], banks=banks
                        ):
                            overflow = False
                            children_to_add[e][i] = True

                # if none of the children fit, add the current node to the accepted list
                if overflow:
                    add = True
            if add:
                best_nodes.append(node)

            for k, v in children_to_add.items():
                child_node = children[k]
                if tuple(child_node.items()) not in visited:
                    visited.add(tuple(child_node.items()))
                    next_graphs = [graphs[i] for i, x in enumerate(v) if x]
                    if next_graphs:
                        if (not prior) or (self.pass_prior(c, factors, prior)):
                            nodes_to_visit.append((child_node, next_graphs))

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        return best_nodes

    # mems: list of mems
    # split: False
    # thres: biggest mem

    # return: List((dict, int))
    # dict: inner bounds of the "good" tiles
    # int: smallest mem size to fit "good" tile

    # "good" tiles: tiles that cannot be enlarged without overflowing any given memory
    def bottom_up_mem(self, mems, split, thres, areas=None, start=None, prior=None):
        assert areas or (not split)
        factors = {}

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([dict([(x, 1) for x in self.prob.keys()])])
        else:
            nodes_to_visit = deque([start])

        # create a database to lookup the smallest mem required to fit a given tile size/set of tile sizes
        # the func also decides if the actual lookup is going to be static or
        # dynamic based on the memory list
        db, db_type = self.generate_db(mems)

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()

            # find the smallest mem required for the current tile
            curr_mem = self.get_mem_size(node, db, db_type=db_type, split=split)

            overflow_curr = True
            for _, c in self.enumerate_children(node, factors).items():

                # find the smallest mem for the child tile
                # if child exceeds the largest mem available, False will be returned
                new_mem = self.get_mem_size(c, db, db_type=db_type, split=split)

                if new_mem:
                    # check if the new mem is valid:
                    #   - if split mem is assumed, check area
                    #   - if unified mem, check depth
                    valid_mem = (
                        self.valid_mem_config_area(new_mem, areas, thres)
                        if split
                        else self.valid_mem_config_depth(new_mem, thres)
                    )
                else:
                    valid_mem = False

                if valid_mem:
                    # if child's mem is the same as the parent's, the parent is not the best node
                    if new_mem == curr_mem:
                        overflow_curr = False

                    # keep traversing the children if they fit
                    if tuple(c.items()) not in visited:
                        visited.add(tuple(c.items()))
                        if (not prior) or (self.pass_prior(c, factors, prior)):
                            nodes_to_visit.append(c)

            # if none of the children fit, add the current node to the accepted
            # list
            if overflow_curr:
                best_nodes.append(
                    (node, self.get_mem_size(node, db, db_type=db_type, split=split))
                )

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        # return [self.convert_node_to_pair(n) for n in best_nodes]
        return best_nodes

    def bottom_up(self, mem, prior=None, start=None, banks=None, bypass=None):
        """
        Construct and traverse the graph starting with packing the memory
        with tensors of size 1, and returning the list of tiles that
        completely fill the memory. This is recommended for small memory
        sizes.
        """
        factors = {}

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([dict([(x, 1) for x in self.prob.keys()])])
        else:
            nodes_to_visit = deque([start])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = [dict([(x, 1) for x in self.prob.keys()])]
        visited = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()

            overflow = True

            children = self.enumerate_children(node, factors).items()

            for _, c in children:
                if self.node_fits(c, mem, banks=banks, bypass=bypass):
                    overflow = False

                    # keep traversing the children if they fit
                    if tuple(c.items()) not in visited:
                        visited.add(tuple(c.items()))
                        if (not prior) or (self.pass_prior(c, factors, prior)):
                            nodes_to_visit.append(c)

            # if none of the children fit, add the current node to the accepted
            # list
            if overflow:
                best_nodes.append(node)

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        # return [self.convert_node_to_pair(n) for n in best_nodes]
        if not best_nodes:
            best_nodes.append(dict([(x, 1) for x in self.prob.keys()]))
        bst_nds = []

        for node in best_nodes:
            if self.node_fits(node, mem, banks=None, bypass=bypass):
                bst_nds.append(node)
            #else:
            #    print("filtering", node)
        return bst_nds

    def bottom_up_next(
        self, mem, next_lev, spatial, prior=None, start=None, banks=None, bypass=None
    ):
        """
        Construct and traverse the graph starting with packing the memory
        with tensors of size 1, and returning the list of tiles that
        completely fill the memory. This is recommended for small memory
        sizes.
        """
        factors = {}

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([dict([(x, 1) for x in self.prob.keys()])])
        else:
            nodes_to_visit = deque([start])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = [dict([(x, 1) for x in self.prob.keys()])]
        visited = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()

            overflow = True

            children = self.enumerate_children(node, factors).items()

            for _, c in children:
                next_lv_tile = c.copy()
                for k, v in c.items():
                    next_lv_tile[k] = v * spatial[0][k] * spatial[1][k]
                if self.node_fits(
                    c, mem, banks=banks, bypass=bypass
                ) and self.node_fits(next_lv_tile, next_lev, banks=banks):
                    overflow = False

                    # keep traversing the children if they fit
                    if tuple(c.items()) not in visited:
                        visited.add(tuple(c.items()))
                        if (not prior) or (self.pass_prior(c, factors, prior)):
                            nodes_to_visit.append(c)

            # if none of the children fit, add the current node to the accepted
            # list
            if overflow:
                best_nodes.append(node)

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        # return [self.convert_node_to_pair(n) for n in best_nodes]
        if not best_nodes:
            best_nodes.append(dict([(x, 1) for x in self.prob.keys()]))
        return best_nodes

    def bottom_up_thres(self, mem, thres, start=None):
        """
        Construct and traverse the graph starting with packing the memory
        with tensors of size 1, and returning the list of tiles that
        completely fill the memory. This is recommended for small memory
        sizes.
        """
        factors = {}

        # enumerate the factors of each dimension in ascending order
        for k, v in self.prob.items():
            factors[k] = [x[0] for x in self.factors(v)]

        # first node to visit is the one that only fits one unit of each tensor
        # nodes are represented as tuples
        if start is None:
            nodes_to_visit = deque([dict([(x, 1) for x in self.prob.keys()])])
        else:
            nodes_to_visit = deque([start])

        # keep track of accepted nodes
        # a node is "accepted" if extending any dimension to the next factor no
        # longer fits (in other words, all its children do not fit)
        best_nodes = []
        visited = set()

        while nodes_to_visit:
            node = nodes_to_visit.popleft()

            overflow = True
            for _, c in self.enumerate_children(node, factors).items():
                if self.node_fits(c, mem):
                    overflow = False

                    # keep traversing the children if they fit
                    if tuple(c.items()) not in visited:
                        visited.add(tuple(c.items()))
                        nodes_to_visit.append(c)

            # if none of the children fit, add the current node to the accepted
            # list
            if type(mem) is tuple or type(mem) is list:
                total_mem = sum(mem)
            else:
                total_mem = mem
            if sum(self.get_size(node)) >= total_mem * thres:
                best_nodes.append(node)

        # format node to be pairs of loop bounds instead of just the inner-most
        # loop bounds
        # return [self.convert_node_to_pair(n) for n in best_nodes]
        return best_nodes

    def enumerate_children(self, node, factors):
        """
        Enumerates the childen of a given node, based on a dictionary of
        factors

        Parameters
        ----------
        node:
            tuple containing the different tiling dimensions
        factors:
            dictionary containing (bound - list) pairs, where the list contains
            factors of the given bound in the order of traversal (e.g ascending
            order for bottom-up)
        """
        children = {}

        # the dimension order in the tuple is PQRSCMN
        # the number of children is <= 7 (less if certain bounds are already at
        # the last factor)
        for b in factors.keys():

            # factors[b])
            # print(node[b])
            index = factors[b].index(node[b])

            if index != (len(factors[b]) - 1):

                # shallow copy is good enough here
                # vals are all primitive
                new_node = node.copy()
                new_node[b] = factors[b][index + 1]
                children[b] = new_node
        return children

    def pass_prior(self, node, factors, prior):
        prior_maxed = True
        prior_bounds = 1
        other_bounds = 1
        for p in prior:
            if node[p] != factors[p][-1]:
                prior_maxed = False
        if prior_maxed:
            return True
        for b in factors.keys():
            if b in prior:
                prior_bounds *= node[b]
            else:
                other_bounds *= node[b]
        return prior_bounds >= other_bounds

    def enumerate_children_prior(self, node, factors, prior):
        """
        Enumerates the childen of a given node, based on a dictionary of
        factors

        Parameters
        ----------
        node:
            tuple containing the different tiling dimensions
        factors:
            dictionary containing (bound - list) pairs, where the list contains
            factors of the given bound in the order of traversal (e.g ascending
            order for bottom-up)
        """
        children = {}

        full = True
        skip = 0
        for b in prior:
            index = factors[b].index(node[b])

            if index != (len(factors[b]) - 1):
                full = False
                break

        # the dimension order in the tuple is PQRSCMN
        # the number of children is <= 7 (less if certain bounds are already at
        # the last factor)
        for b in factors.keys():
            index = factors[b].index(node[b])

            if index != (len(factors[b]) - 1):

                # shallow copy is good enough here
                # vals are all primitive
                new_node = node.copy()
                new_node[b] = factors[b][index + 1]

                prioritized_fact = 1
                rest_fact = 1

                for bb in factors.keys():
                    if bb in prior:
                        prioritized_fact *= new_node[bb]
                    else:
                        rest_fact *= new_node[bb]

                if full or (prioritized_fact > rest_fact):
                    children[b] = new_node
                else:
                    skip += 1
        return children, skip

    def factors(self, dim):
        """
        Enumerates the factors of a given dimension. Each factor is
        represented as a tuple
        """
        return [(x, int(dim / x)) for x in range(1, dim + 1) if dim % x == 0]

    def node_fits(self, tile, mem, banks=None, bypass=None):
        if type(mem) is tuple:
            for i, x in enumerate(zip(self.get_size(tile), mem)):
                if (not bypass or not bypass[i]) and x[0] > x[1]:
                    return False
            return True
        else:
            if banks:
                return (
                    sum([math.ceil(x / (mem / banks)) for x in self.get_size(tile)])
                    <= banks
                )
            else:
                if not bypass:
                    return sum(self.get_size(tile)) <= mem
                else:
                    #print("\n\n")
                    #print(tile)
                    #print("bypass", bypass)
                    #print(self.get_size(tile))
                    return (
                        sum(
                            [
                                (not x[0]) * x[1]
                                for x in zip(bypass, self.get_size(tile))
                            ]
                        )
                        <= mem
                    )

    def node_fits_next(self, tile, mem, mult, banks=None, bypass=None):
        # TODO
        if type(mem) is tuple:
            for i, x in enumerate(zip(self.get_size(tile), mem)):
                if (not bypass or not bypass[i]) and x[0] > x[1]:
                    return False
            return True
        else:
            if banks:
                return (
                    sum(
                        [
                            math.ceil(x * mult[i] / (mem / banks))
                            for i, x in enumerate(self.get_size(tile))
                        ]
                    )
                    <= banks
                )
            else:
                # TODO
                if not bypass:
                    return sum(self.get_size(tile)) <= mem
                else:
                    return (
                        sum(
                            [
                                (not x[0]) * x[1]
                                for x in zip(bypass, self.get_size(tile))
                            ]
                        )
                        <= mem
                    )

