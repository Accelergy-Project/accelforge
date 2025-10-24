import itertools as it
from functools import reduce
import heapq, copy


class Problem:
    """
    This class describes a convolutional layer problem with 7D loop nest

    Attributes
    ----------
    n : int
      batch size
    c : int
      size of input channel
    m : int
      size of output channel
    r : int
      size of kernel width
    s : int
      size of kernel height
    p : int
      size of fmap width
    q : int
      size of fmap height
    w_stride : int
      horizontal stride
    h_stride : int
      vertical stride
    depthwise : bool
      is the problem depthwise?
    depth : int
      depth for depthwise layer
    ifmap_size : int
      total size of ifmap tensor
    ofmap_size : int
      total size of ofmap tensor
    w_size : int
      total size of weight tensor
    non_zero_dimensions : int
      number of effective dimensions in the problem
    solved : bool
      whether the problem is mapped by an optimizer or not
    optimal_energy : float
      if problem is mapped, what is the energy we get from optimal mapping
    solving_time : float
      if problem is mapped, how long did the mapper took
    founded_mapping : string
      string containing the mapping of problem to the accelerator

    Methods
    -------
    get_ifmap_size()
      returns size of ifmap tensor
    get_ofmap_size()
      returns size of ofmap tensor
    get_w_size()
      returns size of weight tensor
    get_total_size()
      return sum ifmap, ofmap and w tensor sizes
    get_status()
      returns True if the problem is marked as solved
    get_stats()
      returns problem statistics (i.e. solving_time, optimal_energy, ...)
    mark_as_solved(solving_time, optimal_energy, map)
      marks the problem as solved and sets the statistics
    get_layer_as_txt()
      returns the problem as a string (e.g. C=12 M=128 ...)
    get_factor(factor)
      returns the value of specified dimension (e.g. value of C)
    set_factor(factor, value)
      resets the value of a specific factor
    duplicate()
      returns a clone of the problem
    divide(sub_problem)
      divides each dimension of the problem by the value of the same dimension in sub_problem
    """

    def __init__(self, n, c, m, r, s, p, q, depthwise=False, depth=0, duplicate=False):
        """
        Parameetrs
        ----------
        n : int
          batch size
        c : int
          size of input channel
        m : int
          size of output channel
        r : int
          size of kernel width
        s : int
          size of kernel height
        p : int
          size of fmap width
        q : int
          size of fmap height
        w_stride : int
          horizontal stride
        h_stride : int
          vertical stride
        depthwise : bool
          whetrher it is a depthwise layer
        depth : int
          if depthwise is true, what is the depth of layer
        """

        self.dimensions = ["n", "c", "m", "r", "s", "p", "q"]
        self.non_zero_dimonsions = 0
        self.n = n
        self.non_zero_dimonsions += 1 if self.n != 1 else 0
        self.c = c
        self.non_zero_dimonsions += 1 if self.c != 1 else 0
        self.m = m
        self.non_zero_dimonsions += 1 if self.m != 1 else 0
        self.r = r
        self.non_zero_dimonsions += 1 if self.r != 1 else 0
        self.s = s
        self.non_zero_dimonsions += 1 if self.s != 1 else 0
        self.p = p
        self.non_zero_dimonsions += 1 if self.p != 1 else 0
        self.q = q
        self.non_zero_dimonsions += 1 if self.q != 1 else 0
        self.w_stride = 1
        self.h_stride = 1
        self.depthwise = depthwise
        self.depth = depth
        self.duplicate = duplicate

        self.ifmap_size = (
            self.n * self.c * (self.p + self.r - 1) * (self.q + self.s - 1)
        )
        self.ofmap_size = self.n * self.m * self.p * self.q
        self.w_size = self.c * self.m * self.r * self.s

        self.total_macs = self.n * self.m * self.p * self.q * self.c * self.r * self.s

        self.solved = False
        self.optimal_energy = float("inf")
        self.solving_time = 0
        self.founded_mapping = ""
        self.tup = (n, c, m, r, s, p, q)
        self.dict = {"p": p, "q": q, "r": r, "s": s, "c": c, "m": m, "n": n}
        self.upper_dict = {"P": p, "Q": q, "R": r, "S": s, "C": c, "M": m, "N": n}

    def get_ifmap_size(self):
        """
        returns ifmap size of the problem
        """

        return self.ifmap_size

    def get_ofmap_size(self):
        """
        returns ofmap size of the problem
        """
        return self.ofmap_size

    def get_w_size(self):
        """
        returns w size of the problem
        """
        return self.w_size

    def get_total_size(self):
        """
        returns total size of the problem
        """
        return self.ifmap_size + self.ofmap_size + self.w_size

    def get_size(self, dtype=""):
        """
        returns size of dtype of the problem

        Parameters
        ----------
        dtype : str
          one of ifmap, ofmap, w or total

        Returns
        -------
        size of dtype in the problem
        """

        if dtype == "ifmap":
            return self.get_ifmap_size()
        elif dtype == "ofmap":
            return self.get_ofmap_size()
        elif dtype == "w":
            return self.get_w_size()
        else:
            return self.get_total_size()

    def get_status(self):
        """
        returns whether the problem is marked as solved or not

        Returns
        -------
        bool
        """
        return self.solved

    def get_stats(self):
        """
        Returns
        -------
        optimal energy and solving time
        """
        return (self.optimal_energy, self.solving_time)

    def get_dimension_value(self, dim):
        """
        returns value of a specific dimension in the problem

        Parameters
        ----------
        dim : str
          n, c, m ,r ,p or q
        """
        if dim == "n":
            return self.n
        elif dim == "c":
            return self.c
        elif dim == "m":
            return self.m
        elif dim == "r":
            return self.r
        elif dim == "s":
            return self.s
        elif dim == "p":
            return self.p
        elif dim == "q":
            return self.q

    def mark_as_solved(self, solving_time, optimal_energy, map):
        """
        Marks the problem as solved and sets the solver stats

        Parameters
        ----------
        optimal_energy : float
          the energy we get from optimal mapping
        solving_time : float
          how long did the mapper took to find the optimal mapping
        map : string
          string containing the optimal mapping of problem to the accelerator
        """

        self.solved = True
        self.solving_time = solving_time
        self.optimal_energy = optimal_energy
        # if self.depthwise:
        #  self.optimal_energy *= self.depth
        self.founded_mapping = map

    def get_layer_as_txt(self):
        """
        Returns the problem as a single descriptive string
        """

        N = "N:" + str(self.n)
        C = " C:" + str(self.c)
        M = " M:" + str(self.m)
        R = " R:" + str(self.r)
        S = " S:" + str(self.s)
        P = " P:" + str(self.p)
        Q = " Q:" + str(self.q)
        D = " D:" + str(self.depth) if self.depthwise else ""
        return N + C + M + R + S + P + Q + D

    def __str__(self):
        return self.get_layer_as_txt()

    def get_layer_as_txt_effective_dims(self):
        """
        Returns the problem as a single descriptive string
        Excludes dimensions that are 1
        """

        dims = {}
        dims["N"] = self.n
        dims["C"] = self.c
        dims["M"] = self.m
        dims["R"] = self.r
        dims["S"] = self.s
        dims["P"] = self.p
        dims["Q"] = self.q
        txt = ""
        for dim in dims.keys():
            if dims[dim] == 1:
                continue
            txt += dim + ":" + str(dims[dim]) + " "
        return txt

    def get_factor(self, factor):
        """
        gets a specific factor of the problem

        Parameters
        ----------
        factor : str

        Returns
        -------
        value of the factor
        """

        if factor == "n":
            return self.n
        elif factor == "c":
            return self.c
        elif factor == "m":
            return self.m
        elif factor == "r":
            return self.r
        elif factor == "s":
            return self.s
        elif factor == "p":
            return self.p
        elif factor == "q":
            return self.q

    def set_factor(self, factor, value):
        """
        sets a specific factor of the problem

        Parameters
        ----------
        factor : str
        value : int
        """

        if factor == "n":
            self.n = value
        elif factor == "c":
            self.c = value
        elif factor == "m":
            self.m = value
        elif factor == "r":
            self.r = value
        elif factor == "s":
            self.s = value
        elif factor == "p":
            self.p = value
        elif factor == "q":
            self.q = value

    def duplicate(self):
        """
        clones the problem and returns it as a ne w object
        """
        return Problem(
            self.n,
            self.c,
            self.m,
            self.r,
            self.s,
            self.p,
            self.q,
            self.depthwise,
            self.depth,
            self.duplicate,
        )

    def refresh(self):
        self.ifmap_size = (
            self.n * self.c * (self.p + self.r - 1) * (self.q + self.s - 1)
        )
        self.ofmap_size = self.n * self.m * self.p * self.q
        self.w_size = self.c * self.m * self.r * self.s
        self.total_macs = self.n * self.m * self.p * self.q * self.c * self.r * self.s
        self.dict = {
            "p": self.p,
            "q": self.q,
            "r": self.r,
            "s": self.s,
            "c": self.c,
            "m": self.m,
            "n": self.n,
        }
        self.upper_dict = {
            "P": self.p,
            "Q": self.q,
            "R": self.r,
            "S": self.s,
            "C": self.c,
            "M": self.m,
            "N": self.n,
        }

    def divide(self, sub_problem):
        """
        divides each dimension of the problem by the value of the same dimension in sub_problem

        Parameters
        ----------
        sub_problem : Problem
          a problem that current problem will be divided by
        """

        for dimension in self.dimensions:
            new_factor = int(
                self.get_factor(dimension) / sub_problem.get_factor(dimension)
            )
            if new_factor < 1:
                new_factor = 1
            self.set_factor(dimension, new_factor)

    def extract_spatial_dimension(self, dims, values):
        for dim, val in zip(dims, values):
            self.set_factor(dim, int(self.get_factor(dim) / val))
        self.refresh()

    def set_order(self, order):
        self.order = order


class _TreeNode:
    """
    Class implements a node of a Dimension Tree. Each node represents a
    specific subset of the problem bounds with a specific perumtation. Its
    children contains the same subset of bounds in the same permutation (in
    addition to the new outermost bound). While each node represents a subset
    of bounds, it only manages and keeps track of information regarding the
    outermost loop of its represented ordering. This class should be private to
    the DimTree class.

    Attributes
    ----------
    name: str
        name/loop-order of the node
    tier: int
        the tier of the loop-order represented by the node
    tens: set(int)
        the set of tensor ids that are reused by the outmost loop bound
        represented by the node
    children: list(_TreeNode)
        list of child nodes
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name: str
        """
        self.name = name
        self.tier = len(name)
        self.tens = set()
        self.children = []

    def insert(self, name, tens):
        """
        Parameters
        ----------
        name: str
        tens: int
            id of tensor that the loop-ordering reuses
        """
        tier = len(name)

        if tier == self.tier + 1:
            # if child already exists, find it and update
            if name in [x.name for x in self.children]:
                for c in self.children:
                    if c.name == name:
                        c.tens.add(tens)
                        return

            # if not, but it should be the child of the curr node, create new
            # child
            new_node = _TreeNode(name)
            new_node.tens.add(tens)
            self.children.append(new_node)
            return

        # if curr node is not the parent of inserted node, find the parent's
        # ancestor using prefix
        for c in self.children:
            if name[: len(c.name)] == c.name:
                c.tens.add(tens)
                c.insert(name, tens)
                return

        # if ancestor does not exist, create them on the fly and recursively
        # insert child
        new_node = _TreeNode(name[: self.tier + 1])
        new_node.tens.add(tens)
        self.children.append(new_node)
        new_node.insert(name, tens)

    def node_is_subset(self, subset_node, node):
        """
        Function checks to see if one node is a subset of another. Node A is a subset of node B if the reused tensors at each loop of A is a subset of the reused tensors at the same loop of B

        Parameters
        ----------
        subset_node, node: dict(str : set(int))
            dictionary that indicates which loops allows the reuse of which sets of tensors

        Returns
        ------
        Bool: Whether subset_node is a subset of node
        """
        # if checked node has bounds not in the other node, we know for sure it
        # is not a subset
        if not set(subset_node.keys()).issubset(set(node.keys())):
            return False

        # for every bound in checked node, check if there are any dimensions
        # that reuse tensors that the other node does not reuse
        for k in subset_node.keys():
            if not subset_node[k].issubset(node[k]):
                return False
        return True

    def find_unique(self):
        """
        Function compares its children and returns the set of unique loop-bounds (every loop-bound should not be a subset of any other loop-bound)

        Returns
        ------
        List((str, dict(str : set(int)))):
            Each node is represented as a str-dict tuple. The string is the name of the node, and the dict is a mapping between each loop bound and the set of tensor ids that each loop bound reuses.
        """
        # just return current node and dict that maps the outermost loop to the
        # tensors that it reuses if the node is a leaf
        if not self.children:
            if self.name:
                return [(self.name, {self.name[-1]: self.tens})]
            else:
                return []

        all_unique = []
        all_unique_reduced = []

        # get the reduced orderings from the children
        for c in self.children:
            unique = c.find_unique()

            # update each child with information of the current node (which
            # tensors are reused by this node's loop dimension?)
            if self.tier > 0:
                # u are in the form (loop name, dict)
                for u in unique:
                    u[1][self.name[-1]] = self.tens

            # keep track of every unique ordering from each child
            all_unique += unique

        # keep track of which node is a subset of at least one other node
        subsets = [False] * len(all_unique)

        # iterate through all pair combinations
        for i, j in it.combinations(range(len(all_unique)), 2):
            # only check pairs that both have not been ruled out as subsets yet
            if not subsets[i] and not subsets[j]:
                if self.node_is_subset(all_unique[i][1], all_unique[j][1]):
                    subsets[i] = True
                # else-if cond here or else pairs of "equivalent" nodes would
                # both get ruled out
                elif self.node_is_subset(all_unique[j][1], all_unique[i][1]):
                    subsets[j] = True

        all_unique_reduced = []
        # only return nodes that are not subsets of at least one other node
        for i, subset in enumerate(subsets):
            if not subset:
                all_unique_reduced.append(all_unique[i])

        return all_unique_reduced

    def __str__(self):
        return f"{self.name} ({self.tens if self.tens else ''})->[{', '.join([str(x) for x in self.children])}]"


class DimTree:
    """
    Class represents a Dimension Tree. This class is essentially a wrapper for the TreeNode class, and only keeps track of the root of the tree.
    """

    def __init__(self, num_tens=None):
        self.root = _TreeNode("")

    def insert(self, name, tens):
        self.root.insert(name, tens)

    def find_unique(self):
        return self.root.find_unique()

    def __str__(self):
        return f"{self.root}"


class GenericProblem:
    """
    Class represents a generic tensor problem. Upon creation, analysis is run to determine which orderings are promising (promotes good reuse).

    Attributes
    ----------
    bounds: Dict(str : int)
        bounds/dimensions of the problem
    tens_desc: List(List((str, str) | str))
        Descriptions of each tensor in the problem. Each tensor's index in the list implicitly determines the tensor's id. Each tensor is described using a list of strings or pair of strings, each indicating the relevant bounds of each dimension (a pair of bounds represents the "sliding window"). The last tensor described is assumed to be the output.
    full_reuse_dims: List(set(str))
        List that maps each tensor to the set of bounds that fully reuses the given tensor. The ith set indicates the bounds that fully reuses the ith tensor.
    part_reuse_dims: List(set(str))
        List that maps each tensor to the set of bounds that partially reuses the given tensor. The ith set indicates the bounds that partially (via sliding window) the ith tensor.
    order_to_reuse: Dict(str : List(str))
        Dictionary that maps each promising ordering to a list of strings, where the ith item indicates which bounds reuses the ith tensor
    tens_to_reuse: List(set(str))
        List that maps each tensor to the set of possible reuses for every promising ordering
    """

    def __init__(self, bounds, tens_desc, size_fun=None):
        """
        Parameters
        ----------
        bounds: Dict(str : int)
            dictionary mapping each bound in the problem to a concrete value
        tens_desc: List(List((str, str) | str))
            see above
        """
        self.bounds = bounds.copy()
        self.tens_desc = tens_desc
        self.size_fun = size_fun

        self.__get_reuse_dims()

        # try to find redundant/equivalent orderings among orderings that
        # prioritize full reuse of each tensor
        # these "promising" orderings differentiate themselves from other
        # permutations of the same set of loop dimensions by reusing other
        # tensors either fully or via sliding window
        order_trees = [DimTree() for _ in self.full_reuse_dims]

        # TODO change bottom to use this tree instead (gives identical results but much cleaner)
        # ord_tree = DimTree()
        # for i, dims in enumerate(self.full_reuse_dims):
        #    if self.part_reuse_dims[i]:
        #        for p in self.part_reuse_dims[i]:
        #            for pp in it.permutations(p, 1):
        #                ord_tree.insert(pp, i)

        #            for k in range(1, len(dims) + 1):
        #                for perm in it.permutations(dims, k):
        #                    l = "".join(perm) + p
        #                    ord_tree.insert(l, i)
        #    else:
        #        for perm in it.permutations(dims):
        #            ord_tree.insert(perm, i)

        for pair in it.permutations(enumerate(self.full_reuse_dims), 2):
            tree_id, tree_dims = pair[0]
            other_id, other_dims = pair[1]

            # find all bounds that allows "sliding" reuse as well as full reuse
            part_intersect = tree_dims.intersection(self.part_reuse_dims[other_id])

            # find all bounds that allow all types of reuse
            intersect = tree_dims.intersection(other_dims)

            # register the partial reuse first
            for p in part_intersect:
                order_trees[tree_id].insert(p, other_id)

            if intersect:
                # any permutation and subset of the full reuse bounds are the
                # innermost loop bounds, followed by any of the partial reuse
                # bounds, presents a unique set of reuse
                if part_intersect:
                    for p in part_intersect:
                        # any subset
                        for k in range(1, len(intersect) + 1):
                            # any permutation of given subset
                            for inter in it.permutations(intersect, k):
                                l = "".join(inter) + p
                                order_trees[tree_id].insert(l, other_id)
                else:
                    # else, any permutation of the full set should be
                    # registered in dim tree
                    for inter in it.permutations(intersect):
                        k = "".join(inter)
                        order_trees[tree_id].insert(k, other_id)

        # find all the orderings that are promising
        # orders is a list of (set(str), List((str, dict(str : set(int))))),
        # or (reuse_bounds, List((prefix, dict(bound : reused_tensors))))
        # reuse_bounds is the set of bounds that allows full reuse of a given
        # tensor
        # prefix is the prefix of the set of loop orderings that reuses various
        # tensors. the loop bounds here should be a subset of reuse_bounds
        # the bound-reused_tensors dict maps each bound in the prefix to the
        # set of tensor that each loop ordering reuses
        orders = [
            (x[0], x[1].find_unique()) for x in zip(self.full_reuse_dims, order_trees)
        ]

        # this maps every promising order to a list of bounds
        # the ith element in the list indicates which bounds reuse the ith
        # tensor under the given promising order
        self.order_to_reuse = {}

        for i, (full, prefixes) in enumerate(orders):
            if self.part_reuse_dims[i]:
                # if partial reuse exists for bound b, ordering b after fulling
                # reusing the tensor is also a promising ordering
                for part in self.part_reuse_dims[i]:
                    if prefixes:
                        # for each promising ordering, append any bound that
                        # gives partial reuse as well
                        for prefix, bounds in prefixes:
                            reuse_factors = ["" for _ in orders]
                            order = prefix + "".join(full - set(prefix)) + part
                            reuse_factors[i] = order

                            # also keep track of which other tensors benefit
                            # from the promising ordering
                            for bound, tens in bounds.items():
                                for t in tens:
                                    reuse_factors[t] += bound
                            self.order_to_reuse[order] = reuse_factors
                    # no prefixes indicates no other tensors can be reused
                    # other than the ith tensor, so just add that to the map
                    else:
                        reuse_factors = ["" for _ in orders]
                        order = "".join(full) + part
                        reuse_factors[i] = order
                        self.order_to_reuse[order] = reuse_factors
            else:
                # same as above, but without having to append the partial reuse
                # bound
                if prefixes:
                    for prefix, bounds in prefixes:
                        reuse_factors = ["" for _ in orders]
                        order = prefix + "".join(full - set(prefix))
                        reuse_factors[i] = order

                        for bound, tens in bounds.items():
                            for t in tens:
                                reuse_factors[t] += bound
                        self.order_to_reuse[order] = reuse_factors
                else:
                    reuse_factors = ["" for _ in orders]
                    order = "".join(full)
                    reuse_factors[i] = order
                    self.order_to_reuse[order] = reuse_factors
        # print(self.order_to_reuse)
        # this maps each tensor to set of possible number of accesses for every
        # promising order
        self.tens_to_reuse = [set() for _ in orders]

        # k is the promising order, v is a list indicating the reuse for each
        # tensor
        for k, v in self.order_to_reuse.items():
            for i, vv in enumerate(v):
                self.tens_to_reuse[i].add(vv)

        self.static = set()
        self.order_to_static = {}
        self.static_to_order = {}

        self.tens_to_ord = {}
        self.ord_to_tens = {}

        for k in self.order_to_reuse.keys():
            self.ord_to_tens[k] = []
            for i, reuse in enumerate(self.full_reuse_dims):
                if set(reuse).issubset(k):
                    self.ord_to_tens[k].append(i)

        for k, v in self.ord_to_tens.items():
            for vv in v:
                if vv in self.tens_to_ord:
                    self.tens_to_ord[vv].append(k)
                else:
                    self.tens_to_ord[vv] = [k]

        for k, v in self.order_to_reuse.items():
            blacklist = set()
            # (tens_id, reuse of <tens_id> given k)
            for i, vv in enumerate(v):
                # bounds that cannot be static are the ones that could be full/part reuse, but are not actually reused due to ordering
                blacklist = blacklist.union(
                    self.full_reuse_dims[i].union(self.part_reuse_dims[i]) - set(vv)
                )
            pot_static = set(k) - blacklist
            self.static.add("".join(pot_static))
            self.order_to_static[k] = "".join(pot_static)
            self.static_to_order["".join(pot_static)] = k

        self.tens_to_sp1 = [set() for _ in range(len(self.tens_desc))]
        self.tens_to_sp2 = [set() for _ in range(len(self.tens_desc))]

        for pair in it.permutations(range(len(self.tens_desc)), 2):
            for i in range(len(self.tens_desc)):
                if i != pair[0] and i != pair[1]:
                    b0 = "".join(
                        self.full_reuse_dims[pair[0]] - self.full_reuse_dims[i]
                    )
                    b1 = "".join(
                        self.full_reuse_dims[pair[1]] - self.full_reuse_dims[i]
                    )
                    b = "".join(set(b0 + b1))
                    self.tens_to_sp2[i].add((b, b))
                    # self.tens_to_sp2[i].add((b0, b1))
                    self.tens_to_sp1[i].add(b0)
                    self.tens_to_sp1[i].add(b1)
        #print(self.tens_to_sp2)

        self.order_to_spatial = {}
        for order in self.order_to_reuse.keys():
            self.order_to_spatial[order] = set()
            tens = []
            for i, reuse_dims in enumerate(self.full_reuse_dims):
                if set(order) == reuse_dims or set(order[:-1]) == reuse_dims:
                    tens.append(i)

            for ten in tens:
                for pairs in it.permutations(self.tens_ind_dims[ten], 2):
                    reused0 = self.reused_tensors(pairs[0])
                    reused1 = self.reused_tensors(pairs[1])

                    if len(reused0.union(reused1)) >= 2:
                        self.order_to_spatial[order].add("".join(pairs))

        self.spatial_to_order = {}

        for k, v in self.order_to_spatial.items():
            for vv in v:
                if vv in self.spatial_to_order:
                    self.spatial_to_order[vv].append(k)
                else:
                    self.spatial_to_order[vv] = [k]

        return

    def reused_tensors(self, bound):
        ret = set()

        for i, reuse in enumerate(self.full_reuse_dims):
            if bound in reuse:
                ret.add(i)
        return ret

    def __get_reuse_dims(self):
        """
        This function populates the full and partial reuse dimension lists (should be private to this class)
        """
        self.full_reuse_dims = []
        self.part_reuse_dims = []
        self.tens_ind_dims = []
        self.part_reuse_sets = []

        for i, tens in enumerate(self.tens_desc):
            tens_dims = set()
            part_reuse_dims = set()
            self.part_reuse_sets.append({})

            # for every tensor dimension, keep track of potential partial reuse
            # (via sliding window), and keep track of relevant dimensions
            for dim in tens:
                if type(dim) is tuple:
                    for d in dim:
                        tens_dims.add(d[-1])
                        part_reuse_dims.add(d[-1])
                        self.part_reuse_sets[i][d[-1]] = tuple([x[-1] for x in dim])
                else:
                    tens_dims.add(dim[-1])
            # full reuse are irrelevant dimensions
            self.tens_ind_dims.append(tens_dims.union(part_reuse_dims))
            self.full_reuse_dims.append(self.bounds.keys() - tens_dims)
            self.part_reuse_dims.append(part_reuse_dims)

    def get_operand_accesses(self):
        num_macs = 1
        for _, v in self.bounds.items():
            num_macs *= v
        ret = [num_macs] * (len(self.tens_desc) - 1)
        out_size = self.get_size(self.bounds, len(self.tens_desc) - 1)
        ret.append((num_macs, num_macs - out_size))
        return ret

    def get_prob_size(self):
        ret = []
        for tens, _ in enumerate(self.tens_desc):
            ret.append(self.get_size(self.bounds, tens))
        return ret

    def get_size(self, dims, tens_id):
        """
        Function returns the size of a given tensor tile based on the dimensions of the given tile level

        Parameters
        ----------
        dims: Dict(str : int)
            dimensions of the given tile level (if tile level is not the innermost, it should be the product of across each level for each dimension)
        tens_id: int
            the id of the tensor whose size is to be determined

        Returns
        -------
        Int
        """

        size = 1
        for dim in self.tens_desc[tens_id]:
            if type(dim) is tuple:
                d = 0
                for dd in dim:
                    if len(dd) > 1:
                        d += int(dd[0]) * (dims[dd[-1]] - 1)
                    else:
                        d += dims[dd] - 1
                d += 1
                size *= d
            else:
                if len(dim) > 1:
                    size *= int(dim[0]) * (dims[dim[-1]] - 1) + 1
                else:
                    size *= dims[dim]
            # d = sum([dims[x] for x in dim]) - (len(dim) > 1)
            # size *= d
        return size

    def __str__(self):
        ret = f"{self.bounds}\n"

        for i, tens in enumerate(self.tens_desc):
            ret += f"{i}: {tens}\n"
            ret += f"  Full Reuse:{self.full_reuse_dims[i]}\n"
            ret += f"  Part Reuse:{self.part_reuse_dims[i]}\n"
        ret += f"  Meaningful Orders: {self.order_to_reuse}, {self.tens_to_reuse}\n"
        ret += f"  Static: {self.static}\n"
        ret += f"  Spatial: {self.order_to_spatial},{self.spatial_to_order}\n"
        ret += f"  Tens to sp: {self.tens_to_sp1},{self.tens_to_sp2}\n"

        return ret


class GenericTile:
    """
    Class represents a valid tiling.

    Attributes
    ----------
    prob: GenericProblem()
        problem that is being tiled
    subtiles: List(dict(str : int))
        A list of dicts, where each dict represents the loop bounds of a particular tiling level
    """

    def __init__(self, prob, tiles=[], spatial=[], spatial_split={}):
        """
        Parameters
        ----------
        prob: GenericProblem
            problem to be tiled
        tiles: List(dict(str : int))
            list of predetermined subtiles
        """
        self.prob = prob
        self.spatial_split = spatial_split.copy()

        # if there are predetermined subtiles, populate subtiles field with
        # those
        if tiles:
            self.subtiles = []
            self.spatial = []
            for i, t in enumerate(tiles):
                # merge spatial that are one after the other
                # also keep track of the original split
                if i < len(tiles) - 1 and spatial[i] and spatial[i + 1]:
                    merged = t.copy()
                    for k, v in t.items():
                        merged[k] *= tiles[i + 1][k]
                    self.subtiles.append(merged)
                    self.spatial_split[i] = t.copy()
                    self.spatial.append(spatial[i])

                elif i == 0 or (not spatial[i]) or (not spatial[i - 1]):
                    self.subtiles.append(t.copy())
                    self.spatial.append(spatial[i])
        # else, initialize with single tile level containing whole problem
        else:
            self.subtiles = [prob.bounds.copy()]
            self.spatial = [False]

    def split(self, bounds, spatial=False, pred=True, tail=True):
        """
        Function returns a new GenericTile object with new a new level of tiling

        Parameters
        ----------
        bounds: Dict(str : int)
            new level of tiling that is extracted from the current tiling
        pred: Bool
            Whether the bounds indicate the new tile to be the inner (T) or other (F) level relative to to split tile level
        tail: Bool
            Whether to split the inner-most (T) or other-most level (F)

        Returns
        -------
        GenericTile that is identical to self, but with one of the tile levels further split into two tile levels
        """
        subtile = {}

        # find the tile level to be split
        split_tile = self.subtiles[-1] if tail else self.subtiles[0]
        index = (len(self.subtiles) - 1) if tail else 0

        # split the subtile into two
        for k, v in bounds.items():
            subtile[k] = int(split_tile[k] / v)

        # determine the position of the new subtile
        if pred:
            new_tiles = [bounds, subtile]
            new_spatial = [spatial, self.spatial[index]]
        else:
            new_tiles = [subtile, bounds]
            new_spatial = [self.spatial[index], spatial]

        # return new class (making shallow copies should be the class
        # constructor's responsibility)
        if tail:
            return GenericTile(
                self.prob,
                self.subtiles[:-1] + new_tiles,
                spatial=self.spatial[:-1] + new_spatial,
                spatial_split=self.spatial_split,
            )
        else:
            return GenericTile(
                self.prob,
                new_tiles + self.subtiles[1:],
                spatial=new_spatial + self.spatial[1:],
                spatial_split=self.spatial_split,
            )

    def __str__(self):
        """
        Function returns formatted string
        """
        ret = ""
        for i, x in enumerate(zip(self.subtiles, self.spatial)):
            ret += f"{x[0]} {'S' if x[1] else 'T'}"
            if i in self.spatial_split:
                ret += f" (=> {self.spatial_split[i]})"
            ret += "\n"
        return ret

    def tup(self, subtile, order):
        """
        Function returns tuple version of specified subtile in specified order (for debugging)
        """
        return tuple([self.subtiles[subtile][k] for k in order])

    def yaml(self, orders, mem_hier, tens_names=[], bypass=None, uneven=None):
        """
        Function returns dictionary that can be directly dumped into the yaml mapping specification file for Timeloop

        Parameters
        ----------
        orders: [str, ...]
            list of strings, where the ith string represents the loop order of the ith memory. The loop order only contains the bounds relevant to reuse (assumes the innermost order is not included)
        mem_hier: [(str, bool), ...]
            list of string-boolean pairs. The ith string represents the name of the ith memory, and the ith boolean indicates whether the ith memory is split
        tens_names: [str, ...]
            list of tensor names (for spatial unrolling and op buffers)
        """
        ret = {"mapping": []}
        mem_i = 0
        default_ord = "".join(self.prob.bounds.keys())

        orders = [default_ord] + orders
        full_orders = []

        # for each order, find the other loop bounds that are not relevant to reuse and pad each with those loop bounds
        for o in orders:
            missing = self.prob.bounds.keys() - set(o)
            full_o = o + "".join(missing)
            full_orders.append(full_o)

        sptl_lvl = 0
        for i, t in enumerate(self.subtiles):
            if self.spatial[i]:
                sptl_lvl += 1
                # 2D case
                if i in self.spatial_split:
                    x_dims = self.spatial_split[i]
                    y_dims = {}
                    for k, v in t.items():
                        y_dims[k] = int(v / x_dims[k])

                    curr_dict = {}
                    curr_dict["target"] = "dummy_buffer" + str(sptl_lvl) if sptl_lvl != 1 else "dummy_buffer"
                    curr_dict["type"] = "temporal"
                    curr_dict["factors"] = " ".join([f"{k}=1" for k in t.keys()])
                    curr_dict["permutation"] = default_ord
                    ret["mapping"].append(curr_dict)

                    curr_dict = {}
                    curr_dict["target"] = "dummy_buffer" + str(sptl_lvl) if sptl_lvl != 1 else "dummy_buffer"
                    curr_dict["type"] = "bypass"
                    curr_dict["bypass"] = tens_names
                    ret["mapping"].append(curr_dict)

                    curr_dict = {}
                    curr_dict["target"] = "dummy_buffer" + str(sptl_lvl) if sptl_lvl != 1 else "dummy_buffer"
                    curr_dict["type"] = "spatial"
                    curr_dict["factors"] = " ".join(
                        [f"{k}={v}" for k, v in y_dims.items()]
                    )
                    curr_dict["permutation"] = default_ord
                    ret["mapping"].append(curr_dict)
                # 1D case
                else:
                    x_dims = t

                curr_dict = {}
                curr_dict["target"] = mem_hier[mem_i][0]
                curr_dict["type"] = "spatial"
                curr_dict["factors"] = " ".join([f"{k}={v}" for k, v in x_dims.items()])
                curr_dict["permutation"] = default_ord
                ret["mapping"].append(curr_dict)

            elif mem_hier[mem_i][1]:                
                for j, tn in enumerate(tens_names):
                    fctrs = {}
                    for k in t:
                        if j == len(self.prob.tens_desc) - 1:
                            fctrs[k] = t[k]
                        else:
                            fctrs[k] = 1

                    if uneven:
                        uneven_tnsr = uneven[mem_i][j]
                        for k in fctrs:
                            fctrs[k] *= uneven_tnsr[k]

                    curr_dict = {}
                    curr_dict["target"] = f"{mem_hier[mem_i][0]}_{tn}"
                    curr_dict["type"] = "temporal"

                        
                    fact_str = " ".join([f"{k}={v}" for k, v in fctrs.items()])
                    curr_dict["factors"] = fact_str
                    curr_dict["permutation"] = full_orders[mem_i]
                    ret["mapping"].append(curr_dict)

                    curr_dict = {}
                    curr_dict["target"] = f"{mem_hier[mem_i][0]}_{tn}"
                    curr_dict["type"] = "bypass"
                    curr_dict["bypass"] = tens_names[0:j] + tens_names[j + 1 :]
                    ret["mapping"].append(curr_dict)
                mem_i += 1
            else:
                curr_dict = {}
                curr_dict["target"] = mem_hier[mem_i][0]
                curr_dict["type"] = "temporal"
                curr_dict["factors"] = " ".join([f"{k}={v}" for k, v in t.items()])
                curr_dict["permutation"] = full_orders[mem_i]
                ret["mapping"].append(curr_dict)
                mem_i += 1
        if bypass:
            for i, b in enumerate(bypass):
                curr_dict = {"target": mem_hier[i][0], "type": "bypass", "bypass": []}
                for tens, bb in enumerate(b):
                    if bb:
                        curr_dict["bypass"].append(tens_names[tens])
                if curr_dict["bypass"]:
                    ret["mapping"].append(curr_dict)
        return ret

    def get_size(self, dims, tens_id):
        """
        Function returns the size of a given tensor tile based on the dimensions of the given tile level

        Parameters
        ----------
        dims: Dict(str : int)
            dimensions of the given tile level (if tile level is not the innermost, it should be the product of across each level for each dimension)
        tens_id: int
            the id of the tensor whose size is to be determined

        Returns
        -------
        Int
        """

        size = 1
        for dim in self.prob.tens_desc[tens_id]:
            if type(dim) is tuple:
                d = 0
                for dd in dim:
                    if len(dd) > 1:
                        d += int(dd[0]) * (dims[dd[-1]] - 1)
                    else:
                        d += dims[dd] - 1
                d += 1
                size *= d
            else:
                if len(dim) > 1:
                    size *= int(dim[0]) * (dims[dim[-1]] - 1) + 1
                else:
                    size *= dims[dim]

            # d = sum([dims[x] for x in dim]) - (len(dim) > 1)
            # size *= d
        return size

    def get_sizes(self, dims):
        """
        Function returns the size of a every tensor tile based on the dimensions of the given tile level

        Parameters
        ----------
        dims: Dict(str : int)
            dimensions of the given tile level (if tile level is not the innermost, it should be the product of across each level for each dimension)

        Returns
        -------
        List(Int)
            the ith element is the size of the ith tensor tile
        """
        sizes = []
        for i in range(len(self.prob.tens_desc)):
            sizes.append(self.get_size(dims, i))
        return sizes

    def get_dual_factors(self):
        """
        Function returns two set of bounds for each tile level -  the product of every inner level including the given tile level, and the product of out level above the given tile level. The inner level bounds are used for tensor size calculations, and the outer level bounds are used to calculate the number of accesses

        Returns
        -------
        List((dict(str : int), dict(str : int)))
        """
        assert len(self.subtiles) >= 2

        ret = []

        # first calculate dual factors for first level
        # inner factors are just the subtile itself
        outer_bounds = dict([(x, 1) for x in self.prob.bounds.keys()])

        # collect outer factors by multiplying each tile level
        for subtile in self.subtiles[1:]:
            for k, v in subtile.items():
                outer_bounds[k] *= v

        ret.append((self.subtiles[0], outer_bounds))

        # for every other tile level (excluding the last), just take the dual factors of the inner level, and scale up the inner factors and scale down the outer factors by the current tile level bounds
        for subtile in self.subtiles[1:-1]:
            inner_bounds = {}
            outer_bounds = {}

            for k, v in subtile.items():
                inner_bounds[k] = ret[-1][0][k] * v
                outer_bounds[k] = int(ret[-1][1][k] / v)
            ret.append((inner_bounds, outer_bounds))
        return ret

    def get_factor_prod(self, bounds, factors):
        """
        Function returns the product of a subset of factors

        Parameters
        ----------
        bounds: str
            list of bounds to be included in the product
        factors: dict(str : int)
            actual values of each bound

        Returns
        -------
        int
        """
        prod = 1
        for b in bounds:
            prod *= factors[b]
        return prod

    def get_mem_sizes(self, mems, split):
        """
        Function returns the smallest memory required to hold each tensor at each memory level.

        Parameters
        ----------
        mems: [[int, ...], ...]
           The ith int list indicates the list of available memory sizes at the ith memory level.
        split: [bool, ...]
            The ith boolean indicates whether the ith memory level is split or or not (unified)

        Returns
        -------
        [(int, ...), ...]
            The jth int of the ith tuple indicates the smallest memory size required to hold the jth operand in the ith memory level. If the memory level is unified, every int in the tuple should be the same value.
        """
        ret = []
        mem_i = 0
        for i, factors in enumerate(self.get_dual_factors()):
            if not self.spatial[i]:
                if split[mem_i]:
                    sizes = self.get_sizes(factors[0])
                else:
                    sizes = [sum(self.get_sizes(factors[0]))]

                lev = []
                for size in sizes:
                    # TODO if this is a bottleneck, switch to binary search
                    for m in mems[mem_i]:
                        if size <= m:
                            lev.append(m)
                            break
                if not split[mem_i]:
                    lev = lev * len(self.prob.tens_desc)
                ret.append(tuple(lev))
                mem_i += 1
        return ret

    def get_accesses(self):
        """
        Function returns the total access counts (in words) of the given tiling (split by level and tensor)

        Returns
        -------
        List(dict(str : List(int | (int, int))))
        Each dict maps a promising ordering to a list of ints/(int, int), where the ith entry is the number of accesses of the ith tensor under the given promising ordering. The ith dict in the list corresponds to the ith tiling level.
        """
        ret = []
        mem_lev = 0
        for lev, dual_factors in enumerate(self.get_dual_factors()):
            # skip spatial tiles (only contribute to reuse)
            if not self.spatial[lev]:
                factors = dual_factors[1]
                sizes = self.get_sizes(dual_factors[0])

                # access_maps is a List(Dict(str : int))
                # each dict maps a reuse combination to the number of accesses under that reuse combination for a given tensor, and the ith entry of the list correspond to the ith tensor
                access_maps = []

                # each level is an entry (dict) in the returned list
                ret.append({})

                spatial_next = lev < (len(self.subtiles) - 1) and self.spatial[lev + 1]
                reuse_subtile = (lev + 2) if spatial_next else (lev + 1)

                non_ones = set()
                for k, v in self.subtiles[reuse_subtile].items():
                    if v > 1:
                        non_ones.add(k)

                for i, tens in enumerate(sizes):
                    spatial_reuse = (
                        self.get_factor_prod(
                            "".join(self.prob.full_reuse_dims[i]),
                            self.subtiles[lev + 1],
                        )
                        if spatial_next
                        else 1
                    )
                    bl_tile_accesses = self.get_factor_prod(
                        "".join(self.prob.bounds.keys()), factors
                    )
                    # the baseline access of each tensor at each level is just the product of all the outer bounds times the tensor tile size
                    no_reuse = tens * bl_tile_accesses
                    access_maps.append({})

                    # for each type of possible reuse, calculate the access by scaling the baseine accesses accordingly
                    for reuse in self.prob.tens_to_reuse[i]:
                        # sliding window reuse is involved
                        if reuse and reuse[-1] in self.prob.part_reuse_dims[i]:
                            have_partial = True

                            if spatial_next:
                                # have_partial = have_partial and (self.subtiles[lev+1][reuse[-1]] == 1)
                                have_partial = have_partial and reduce(
                                    lambda x, y: x and y,
                                    map(
                                        lambda x: self.subtiles[lev + 1][x] == 1,
                                        self.prob.part_reuse_sets[i][reuse[-1]],
                                    ),
                                )
                            if have_partial:
                                # update the bounds to include reuse dim
                                part_dim = reuse[-1]
                                new_bounds = dual_factors[0].copy()
                                new_bounds[part_dim] *= self.subtiles[reuse_subtile][
                                    part_dim
                                ]

                                # recalculate tile size (remove overlaps)
                                new_size = self.get_size(new_bounds, i)

                                # calculate access with reuse considered
                                with_reuse = int(
                                    new_size
                                    * bl_tile_accesses
                                    / self.get_factor_prod(
                                        reuse, self.subtiles[reuse_subtile]
                                    )
                                )
                            else:
                                with_reuse = no_reuse
                        # all full reuse
                        else:
                            # TODO find more robust way to do this
                            if non_ones.issubset(self.prob.full_reuse_dims[i]):
                                reuse_factor = self.get_factor_prod(
                                    non_ones, self.subtiles[reuse_subtile]
                                )
                            else:
                                reuse_factor = self.get_factor_prod(
                                    reuse, self.subtiles[reuse_subtile]
                                )
                            with_reuse = int(no_reuse / reuse_factor)

                        # for outputs, data is init to zero, so first fill can be ignored
                        if i == (len(sizes) - 1):
                            updates = with_reuse
                            full_tens_size = self.get_size(
                                self.prob.bounds, len(sizes) - 1
                            )
                            access_maps[i][reuse] = (
                                (updates, updates - full_tens_size),
                                (
                                    int(updates / spatial_reuse),
                                    int(updates / spatial_reuse) - full_tens_size,
                                ),
                            )
                        else:
                            access_maps[i][reuse] = (
                                with_reuse,
                                int(with_reuse / spatial_reuse),
                            )

                # finally, for each promising order, see how each tensor is reused to determine number of accesses for each tensor
                # orders is a str (prefix of loop ordering), accesses is list of str, where the ith index is the corresponding reuse bounds for the ith tensor
                for orders, accesses in self.prob.order_to_reuse.items():
                    ret[mem_lev][orders] = [
                        access_maps[i][x] for i, x in enumerate(accesses)
                    ]

                mem_lev += 1

        # this is for the special case where an intermediate (not first or last) tile level has only reuse bounds of one particular tensor and nothing else (unlikely since this suggests underutil, and with spatial, the different levels tend to reuse different tensors)
        spec_tens = {}
        if len(self.subtiles) > 2:
            mem_i = len(self.subtiles) - 2
            # for each intermediate tile levels
            for i in range(len(self.subtiles) - 2, 0, -1):
                if self.spatial[i]:
                    continue
                # find bounds that are not 1 (actually matters)
                non_ones = set(
                    [x for x in self.prob.bounds.keys() if self.subtiles[i][x] != 1]
                )
                for j, reuse_dims in enumerate(self.prob.full_reuse_dims):
                    # if the only non-one bounds reuse the same tensor, a special case exists where the outer and inner layers reusing the same tensor has even more reuse than calculated
                    # an adjustment may be made later while calculating cost
                    if non_ones.issubset(reuse_dims):
                        for order in self.prob.order_to_reuse.keys():
                            if (
                                len(order) >= len(reuse_dims)
                                and set(order[: len(reuse_dims)]) == reuse_dims
                            ):
                                key = order
                                # values are (tensor id, mem level)
                                if key in spec_tens:
                                    spec_tens[key].append((j, mem_i - 1))
                                else:
                                    spec_tens[key] = [(j, mem_i - 1)]
                mem_i -= 1

        # accesses in the special case
        spec_ret = {}
        # handles only single case for now
        return ret, spec_ret
        for inner_order, v in spec_tens.items():
            for tens, mem_lev in v:
                for outer_order in self.prob.tens_to_ord[tens]:
                    if (inner_order, outer_order) not in spec_ret:
                        inner = ret[mem_lev - 1][inner_order].copy()
                        outer = ret[mem_lev][outer_order].copy()
                    else:
                        inner = spec_ret[(inner_order, outer_order)][0].copy()
                        outer = spec_ret[(inner_order, outer_order)][1].copy()

                    if tens == len(self.prob.tens_desc) - 1:
                        spatial = int(inner[tens][0][0] / inner[tens][1][0])
                        inner[tens] = (
                            (outer[tens][0][0] * spatial, outer[tens][0][1]),
                            outer[tens][1],
                        )
                    else:
                        spatial = int(inner[tens][0] / inner[tens][1])
                        inner[tens] = (outer[tens][0] * spatial, outer[tens][1])

                    spec_ret[(inner_order, outer_order)] = [inner, outer]

        return ret, spec_ret

    def get_spatial_count(self):
        ret = 1
        for i, tile in enumerate(self.subtiles):
            if self.spatial[i]:
                for _, v in tile.items():
                    ret *= v
        return ret

    # only for L1 for now
    def get_windows(self, ord):
        macs_per_it = 1
        for b in self.subtiles[0].values():
            macs_per_it *= b

        ret = [macs_per_it] * len(self.prob.tens_desc)

        for i, reuse in enumerate(self.prob.order_to_reuse[ord]):
            for b in reuse:
                if b in self.prob.full_reuse_dims[i]:
                    ret[i] *= self.subtiles[2][b]
        return ret

    # only for L2 for now
    def get_bw(self, ord):
        sizes = self.get_sizes(self.get_dual_factors()[1][0])

        return [x[0] / x[1] for x in zip(sizes, self.get_windows(ord))]

class Dataflow:
    """
    Class for wrapping multiple tiles and a problem
    to represent a specific dataflow.
    An important use-case is getting the correct size of each level,
    that also represents all the sublevels it contains. Normally,
    tiles that describe dataflow at each level does not represent ture
    size of each level (i.e. product of all the lower levels with current size).
    This class can be used to overcome this.

    Attributes
    ----------
    problem : Problem
      the problem dataflow is solving
    tiles: list
      list of tiles that describe each level of dataflow
    adjusted_tile_problems : list
      list of tiles, but each level is adjusted so that factors
      are product of that level factors and all the levels below it.
      This helps to infer the true size of level.
    dataflow_energy : float
      resulting energy of the dataflow. default to zero

    Methods
    -------
    get_size(dtype, level)
      returns true size of dtype in level
    print_dataflow()
      prints the dataflow
    """

    def __init__(self, problem, tiles, dataflow_energy=0):
        self.problem = problem
        self.dataflow_energy = dataflow_energy
        self.tiles = tiles
        self.adjusted_tile_problems = [self.tiles[0].tile_problem]
        for i, tile in enumerate(tiles[1:]):
            low_t = self.adjusted_tile_problems[i]
            up_t_n = tile.tile_problem.n * low_t.n
            up_t_c = tile.tile_problem.c * low_t.c
            up_t_m = tile.tile_problem.m * low_t.m
            up_t_r = tile.tile_problem.r * low_t.r
            up_t_s = tile.tile_problem.s * low_t.s
            up_t_p = tile.tile_problem.p * low_t.p
            up_t_q = tile.tile_problem.q * low_t.q
            adjusted_problem = Problem(
                up_t_n, up_t_c, up_t_m, up_t_r, up_t_s, up_t_p, up_t_q
            )
            self.adjusted_tile_problems.append(adjusted_problem)

    def inject_spatial_factors(self):
        pass

    def get_size(self, dtype, level):
        """
        Returns
        -------
        The true size of dtype at level.
        """

        ifmap_size = self.adjusted_tile_problems[level].get_ifmap_size()
        if dtype == "ifmap":
            return ifmap_size
        ofmap_size = self.adjusted_tile_problems[level].get_ofmap_size()
        if dtype == "ofmap":
            return ofmap_size
        w_size = self.adjusted_tile_problems[level].get_w_size()
        if dtype == "w":
            return w_size
        if dtype == "total":
            return ifmap_size + ofmap_size + w_size

    def get_focus(self, level):
        if self.get_size("ifmap", level) >= self.get_size("w", level) and self.get_size(
            "ifmap", level
        ) >= self.get_size("ofmap", level):
            return "ifmap"
        elif self.get_size("ofmap", level) >= self.get_size("w", level):
            return "ofmap"
        else:
            return "w"

    def print_dataflow(self):
        """
        Prints the dataflow
        """

        print(
            "DRAM (ifmap: "
            + str(self.problem.get_size("ifmap"))
            + "  ofmap: "
            + str(self.problem.get_size("ofmap"))
            + "  w: "
            + str(self.problem.get_size("w"))
            + " )"
        )
        print(self.tiles[-1].upper_factors.get_layer_as_txt())

        for level in range(len(self.tiles), 0, -1):
            print(
                "\nL"
                + str(level)
                + ": ifmap: "
                + str(self.adjusted_tile_problems[level - 1].get_size("ifmap"))
                + "  ofmap: "
                + str(self.adjusted_tile_problems[level - 1].get_size("ofmap"))
                + "  w: "
                + str(self.adjusted_tile_problems[level - 1].get_size("w"))
                + " )"
            )
            print(self.tiles[level - 1].tile_problem.get_layer_as_txt())
        return

    def get_dataflow_as_txt(self):
        lines = []
        lines.append(
            "DRAM (ifmap: "
            + str(self.problem.get_size("ifmap"))
            + "  ofmap: "
            + str(self.problem.get_size("ofmap"))
            + "  w: "
            + str(self.problem.get_size("w"))
            + " )"
        )
        for level in range(len(self.tiles), 0, -1):
            lines.append(
                "\nL"
                + str(level)
                + ": (ifmap: "
                + str(self.adjusted_tile_problems[level - 1].get_size("ifmap"))
                + "  ofmap: "
                + str(self.adjusted_tile_problems[level - 1].get_size("ofmap"))
                + "  w: "
                + str(self.adjusted_tile_problems[level - 1].get_size("w"))
                + " )"
            )
            lines.append(self.tiles[level - 1].tile_problem.get_layer_as_txt())

        return lines


class EnergyBreakdown:
    def __init__(self, total_e, levels_e, mac_e):

        self.total = total_e
        self.levels_energies = [
            {"ifmap": 0.0, "ofmap": 0.0, "w": 0.0} for _ in range(len(levels_e))
        ]
        self.levels_energies_percent = [
            {"ifmap": 0.0, "ofmap": 0.0, "w": 0.0} for _ in range(len(levels_e))
        ]
        for level, level_e in enumerate(levels_e):
            self.levels_energies[level]["ifmap"] = level_e["ifmap"]
            self.levels_energies[level]["ofmap"] = level_e["ofmap"]
            self.levels_energies[level]["w"] = level_e["w"]

            self.levels_energies_percent[level]["ifmap"] = level_e["ifmap"] / total_e
            self.levels_energies_percent[level]["ofmap"] = level_e["ofmap"] / total_e
            self.levels_energies_percent[level]["w"] = level_e["w"] / total_e
        self.MAC = mac_e
        self.MAC_prc = self.MAC / self.total

    def level_energy_prc(self, level):
        return (
            self.levels_energies_percent[level]["ifmap"]
            + self.levels_energies_percent[level]["ofmap"]
            + self.levels_energies_percent[level]["w"]
        )


class AccessBreakdown:
    def __init__(self, levels_accesses):

        self.levels_accesses = [
            {"ifmap": 0.0, "ofmap": 0.0, "w": 0.0} for _ in range(len(levels_accesses))
        ]
        for level, level_access in enumerate(levels_accesses):
            self.levels_accesses[level]["ifmap"] = level_access["ifmap"]
            self.levels_accesses[level]["ofmap"] = level_access["ofmap"]
            self.levels_accesses[level]["w"] = level_access["w"]


class Unrolling:
    def __init__(self, PE_X, PE_Y):
        self.X_axis_dims = []
        self.X_axis_values = []
        self.X_utilized = 1
        self.Y_axis_dims = []
        self.Y_axis_values = []
        self.Y_utilized = 1
        self.total_X = PE_X
        self.total_Y = PE_Y

    def add_X_unrolling(self, dim, value):
        self.X_axis_dims.append(dim)
        self.X_axis_values.append(value)
        self.X_utilized *= value

    def add_Y_unrolling(self, dim, value):
        self.Y_axis_dims.append(dim)
        self.Y_axis_values.append(value)
        self.Y_utilized *= value

    def get_utilized_PEs(self):
        PEs = 1
        for dim, val in self.unified():
            PEs *= val
        return PEs

    def X_unrolling_match(self, value):
        if self.X_utilized < self.total_X and self.X_utilized % value == 0:
            return True
        else:
            return False

    def Y_unrolling_match(self, value):
        if self.Y_utilized < self.total_Y and self.Y_utilized % value == 0:
            return True
        else:
            return False

    def print_unrolling(self):
        print("Unrolling:")
        print("axis 1")
        for x, y in zip(self.X_axis_dims, self.X_axis_values):
            print(x, ":", y)
        print("axis 2")
        for x, y in zip(self.Y_axis_dims, self.Y_axis_values):
            print(x, ":", y)

    def focus(self):
        focus = []
        w_dims = ["c", "m", "r", "s"]
        ifmap_dims = ["c", "n", "p", "q", "r", "s"]
        ofmap_dims = ["m", "n", "p", "q"]
        dims_unrolled = self.X_axis_dims + self.Y_axis_dims
        flag = True
        for dim in dims_unrolled:
            if dim not in w_dims:
                flag = False
                break
        if flag:
            focus.append("w")
        flag = True
        for dim in dims_unrolled:
            if dim not in ifmap_dims:
                flag = False
                break
        if flag:
            focus.append("ifmap")
        flag = True
        for dim in dims_unrolled:
            if dim not in ofmap_dims:
                flag = False
                break
        if flag:
            focus.append("ofmap")
        return focus

    def as_list(self):
        ls = [[], []]
        for i in range(len(self.X_axis_dims)):
            ls[0].append((self.X_axis_dims[i], self.X_axis_values[i]))
        for i in range(len(self.Y_axis_dims)):
            ls[1].append((self.Y_axis_dims[i], self.Y_axis_values[i]))
        return ls

    def unified(self):

        ls_unified_dict = {}
        for axis in self.as_list():
            for dim, val in axis:
                if dim not in ls_unified_dict.keys():
                    ls_unified_dict[dim] = val
                else:
                    ls_unified_dict[dim] *= val
        ls_unified = []
        for dim in ls_unified_dict.keys():
            ls_unified.append((dim, ls_unified_dict[dim]))
        # ls_unified = [x for x in ls[0]] + [x for x in ls[1]]
        return ls_unified

    def as_string(self):
        txt = "X: "
        for i in range(len(self.X_axis_dims)):
            txt += self.X_axis_dims[i] + ":" + str(self.X_axis_values[i]) + " "
        txt += "  Y: "
        for i in range(len(self.Y_axis_dims)):
            txt += self.Y_axis_dims[i] + ":" + str(self.Y_axis_values[i]) + " "
        return txt

    def __str__(self):
        return self.as_string()

    def switch_axes(self):
        tmp_dims = self.X_axis_dims
        tmp_vals = self.X_axis_values
        self.X_axis_dims = self.Y_axis_dims
        self.X_axis_values = self.Y_axis_values
        self.Y_axis_dims = tmp_dims
        self.Y_axis_values = tmp_vals
        tmp = self.total_X
        self.total_X = self.total_Y
        self.total_Y = tmp


class Mapping:
    def __init__(self, problem):
        self.problem = problem
        self.mini_problem = copy.deepcopy(problem)
        self.tile1 = None
        self.tile2 = None
        self.DR_order = []
        self.L2_order = []
        self.L1_order = []
        self.unrolling = None
        self.cost = float("inf")
        self.optimization_time = 0

    def finalize_unrolling(self):
        for dim, val in self.unrolling.unified():
            if dim != "d":
                self.mini_problem.set_factor(
                    dim, int(self.mini_problem.get_factor(dim) / val)
                )
            else:
                self.mini_problem.depth = int(self.mini_problem.depth / val)

    def get_tiles(self):
        return [self.tile1, self.tile2]

    def get_orders(self):
        return {"DR": self.DR_order, "L2": self.L2_order, "L1": self.L1_order}

    def get_permutations(self):
        # for Timeloop simulations
        return [
            "".join(self.L1_order).upper()[::-1],
            "".join(self.L2_order).upper()[::-1],
            "".join(self.DR_order).upper()[::-1],
        ]

    def get_orders_inner_loops(self):
        local_L2_orders = ""
        for c in self.L2_order[-3:]:
            local_L2_orders += c.upper()
        if local_L2_orders[-1] == "M":
            local_L2_orders = local_L2_orders[1:]
        local_DR_orders = ""
        for c in self.DR_order[-3:]:
            local_DR_orders += c.upper()
        if local_DR_orders[-1] == "M":
            local_DR_orders = local_DR_orders[1:]
        return ["", local_L2_orders, local_DR_orders]

    def get_unrolling(self):
        return self.unrolling

    def get_temporally_reused_tensor(self):
        inner_loop_to_reuse = {
            "p": "w",
            "q": "w",
            "n": "w",
            "m": "ifmap",
            "s": "ofmap",
            "r": "ofmap",
            "c": "ofmap",
        }
        return [
            "",
            inner_loop_to_reuse[self.L2_order[-1]],
            inner_loop_to_reuse[self.DR_order[-1]],
        ]

    def set_cost(self, cost):
        self.cost = cost

    def __str__(self):
        assert self.tile2 != None
        my_txt = "\n"
        my_txt += "L1 tile: " + self.tile1.tile_problem.get_layer_as_txt() + "\n"
        if type(self.L2_order) is list:
            my_txt += "L2 order is: " + "".join(self.L2_order) + "\n"
        else:
            my_txt += "L2 order is: " + self.L2_order + "\n"
        my_txt += "L2 tile: " + self.tile2.tile_problem.get_layer_as_txt() + "\n"
        if type(self.DR_order) is list:
            my_txt += "DR order is: " + "".join(self.DR_order) + "\n"
        else:
            my_txt += "DR order is: " + self.DR_order + "\n"
        my_txt += "DR tile: " + self.tile2.upper_factors.get_layer_as_txt() + "\n"
        my_txt += "Unrolling: " + self.unrolling.as_string() + "\n"
        my_txt += "cost: " + str(self.cost) + "\n"
        my_txt += "\n"
        return my_txt

    def duplicate(self):
        mapping_duplicate = Mapping(self.problem)
        mapping_duplicate.mini_problem = self.mini_problem
        mapping_duplicate.tile1 = self.tile1
        mapping_duplicate.tile2 = self.tile2
        mapping_duplicate.DR_order = self.DR_order
        mapping_duplicate.L2_order = self.L2_order
        mapping_duplicate.L1_order = self.L1_order
        mapping_duplicate.unrolling = self.unrolling
        mapping_duplicate.cost = self.cost
        return mapping_duplicate


class Mapping_1level:
    def __init__(self, problem):
        self.problem = problem
        self.mini_problem = problem.duplicate()
        self.tile1 = None
        self.DR_order = []
        self.L1_order = []
        self.unrolling = None
        self.cost = float("inf")

    def finalize_unrolling(self):
        for dim, val in self.unrolling.unified():
            #print(dim)
            if dim != "d":
                self.mini_problem.set_factor(
                    dim, int(self.mini_problem.get_factor(dim) / val)
                )
            else:
                self.mini_problem.depth = int(self.mini_problem.depth / val)

    def get_tiles(self):
        return [self.tile1]

    def get_orders(self):
        return {"DR": self.DR_order, "L1": self.L1_order}

    def get_orders_inner_loops(self):
        local_DR_orders = ""
        for c in self.DR_order[-3:]:
            local_DR_orders += c.upper()
        if local_DR_orders[-1] == "M":
            local_DR_orders = local_DR_orders[1:]
        return ["", local_DR_orders]

    def get_unrolling(self):
        return self.unrolling

    def get_temporally_reused_tensor(self):
        inner_loop_to_reuse = {
            "p": "w",
            "q": "w",
            "n": "w",
            "m": "ifmap",
            "s": "ofmap",
            "r": "ofmap",
            "c": "ofmap",
        }
        return ["", inner_loop_to_reuse[self.DR_order[-1]]]

    def set_cost(self, cost):
        self.cost = cost

    def print_info(self):
        print("")
        print("L1 tile: ", self.tile1.tile_problem.get_layer_as_txt())
        print("Unrolling: ", self.unrolling.as_string())
        print("cost: ", self.cost)
        print("")

    def duplicate(self):
        mapping_duplicate = Mapping_1level(self.problem)
        mapping_duplicate.mini_problem = self.mini_problem
        mapping_duplicate.tile1 = self.tile1
        mapping_duplicate.DR_order = self.DR_order
        mapping_duplicate.L1_order = self.L1_order
        mapping_duplicate.unrolling = self.unrolling
        mapping_duplicate.cost = self.cost
        return mapping_duplicate


class Mapping_DW(Mapping):
    def __init__(self, problem):
        super().__init__(problem)
        self.modes = {"full_PE", "PE_row", "single_PE"}
        self.mode = "full_PE"
        self.groups = problem.depth

    def copy_mapping(self, mapping):
        self.cost = mapping.cost
        self.unrolling = mapping.unrolling
        self.mini_problem = mapping.mini_problem
        self.L1_order = mapping.L1_order
        self.L1_order = mapping.L1_order
        self.L2_order = mapping.L2_order
        self.DR_order = mapping.DR_order
        self.tile1 = mapping.tile1
        self.tile2 = mapping.tile2
        self.optimization_time = mapping.optimization_time
        return

    def print_info(self):
        print("Depthwise execution mode: ", self.mode)
        print("Depth: ", self.groups)
        super().print_info()


class TileDepthwise:
    def __init__(self, n, c, m, r, s, p, q, d, upper_level_problem):
        self.upper_level_problem = upper_level_problem
        self.tile_problem = Problem(n, c, m, r, s, p, q, depth=d, depthwise=True)
        for dim in ["n", "c", "m", "r", "s", "p", "q"]:
            if self.tile_problem.get_factor(dim) == 0:
                self.tile_problem.set_factor(dim, 1)
        if self.tile_problem.depth == 0:
            self.tile_problem.depth = 1
        self.upper_factors = Problem(
            int(self.upper_level_problem.n / self.tile_problem.n),
            int(self.upper_level_problem.c / self.tile_problem.c),
            int(self.upper_level_problem.m / self.tile_problem.m),
            int(self.upper_level_problem.r / self.tile_problem.r),
            int(self.upper_level_problem.s / self.tile_problem.s),
            int(self.upper_level_problem.p / self.tile_problem.p),
            int(self.upper_level_problem.q / self.tile_problem.q),
            depth=int(self.upper_level_problem.depth / self.tile_problem.depth),
            depthwise=True,
        )
