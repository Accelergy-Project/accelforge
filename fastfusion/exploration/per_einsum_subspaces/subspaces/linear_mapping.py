class LinearMapping:
    def __init__(self):
        self.mapping = []

    def __iter__(self):
        return iter(self.mapping)

    def __getitem__(self, key):
        return self.mapping[key]

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return repr(self.mapping)

    def copy(self):
        lm = LinearMapping()
        lm.mapping = self.mapping.copy()
        return lm

    def add_compute(self, einsum_name, target):
        self.mapping.append(
            {"type": "compute", "einsum": einsum_name, "target": target}
        )

    def add_temporal(
        self,
        rank_name,
        tile_shape=None,
        tile_shape_constraint=None,
        factor_constraint=None,
    ):
        node = {"type": "temporal", "rank": rank_name}
        if tile_shape is not None:
            node["tile_shape"] = tile_shape
        if tile_shape_constraint is not None:
            node["tile_constraint"] = tile_shape_constraint
        if factor_constraint is not None:
            node["factor_constraint"] = factor_constraint
        self.mapping.append(node)

    def add_spatial(
        self,
        rank_name,
        tile_shape=None,
        tile_shape_constraint=None,
        factor_constraint=None,
    ):
        node = {"type": "spatial", "rank": rank_name}
        if tile_shape is not None:
            node["tile_shape"] = tile_shape
        if tile_shape_constraint is not None:
            node["tile_constraint"] = tile_shape_constraint
        if factor_constraint is not None:
            node["factor_constraint"] = factor_constraint
        self.mapping.append(node)

    def add_sequential(self, idx=None):
        node = {"type": "sequential"}
        if idx is None:
            self.mapping.append(node)
        else:
            self.mapping.insert(idx, node)

    def add_pipeline(self):
        self.mapping.append({"type": "pipeline"})

    def add_storage(self, target, dspaces, idx=None, exploits_reuse=False):
        node = {"type": "storage",
                "target": target,
                "dspace": dspaces,
                "exploits_reuse": exploits_reuse}
        if idx is None:
            self.mapping.append(node)
        else:
            self.mapping.insert(idx, node)

