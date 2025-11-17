from fastfusion.frontend.specification import Specification
from fastfusion.frontend.workload import EinsumName
from fastfusion.accelerated_imports import pd
from typing import Union
from fastfusion.frontend.workload.workload import TensorName
from fastfusion.mapper.FFM._make_pmappings.make_pmappings import (
    get_num_computes,
    get_per_tensor_size,
)


class Mappings:
    def __init__(
        self,
        spec: Specification,
        einsum_names: list[EinsumName],
        data: pd.DataFrame,
        total_mappings: int,
        valid_mappings: int,
    ):
        self.spec = spec
        self.einsum_names = einsum_names
        self.data = data
        self.total_mappings = total_mappings
        self.valid_mappings = valid_mappings

    def num_computes(self, einsum_name: EinsumName | None = None) -> int:
        # TODO: this is not correct if there are recomputation.
        return get_num_computes(self.spec, einsum_name)

    def per_tensor_size(self) -> dict[TensorName, int]:
        return get_per_tensor_size(self.spec)

    def _update(self, **kwargs):
        data = dict(
            spec=self.spec,
            einsum_names=self.einsum_names,
            data=self.data,
            total_mappings=self.total_mappings,
            valid_mappings=self.valid_mappings,
        )
        data.update(kwargs)
        return Mappings(**data)

    def __getitem__(self, key: str | int) -> Union[pd.Series, "Mappings"]:
        if isinstance(key, int):
            return self._update(data=pd.DataFrame(self.data.iloc[key]).T)
        if len(self) == 1:
            return self.data[key].iloc[0]
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def per_component_energy(self) -> dict[tuple[EinsumName, str, str], float]:
        """
        Returns a dictionary of:
            {(Einsum name, Component name, Action name): Energy}
        """
        per_component_energy = {}
        for col in self.data.columns:
            if col.endswith("component_energy"):
                einsum_name, component, action, *_ = col.split("_")
                per_component_energy[(einsum_name, component, action)] = self.data[col]
        return self.data

    def per_component_latency(self) -> dict[tuple[EinsumName, str], float]:
        """
        Returns a dictionary of:
            {(Einsum name, Component name): Latency}
        """
        per_component_latency = {}
        for col in self.data.columns:
            if col.endswith("latency") and col != "Total_latency":
                einsum_name, component, *_ = col.split("_")
                per_component_latency[(einsum_name, component)] = self.data[col]
        return per_component_latency

    def _get_cols(self, key: str) -> list[str]:
        found_index = None
        found = []
        for col in self.data.columns:
            col = col.split("<SEP>")
            if key not in col:
                continue

            if sum(c == key for c in col) > 1:
                raise ValueError(
                    f"Key {key} found multiple times in the column names. "
                    f'Columns: "{col}"'
                )

            if found_index is not None and col.index(key) != found_index:
                raise ValueError(
                    f"Key {key} found at varying indexes in the column names. "
                    f'Columns: "{col}" and "{found}"'
                )
            found_index = col.index(key)
            found.append("<SEP>".join(col))
        return found

    def access(self, *keys: str) -> "Mappings":
        assert len(set(self.data.columns)) == len(
            self.data.columns
        ), "Columns must be unique"

        if len(keys) != 1:
            for k in keys:
                self = self.access(k)
            return self

        key = keys[0]
        col_renames = {}
        for col in self._get_cols(key):
            col_renames[col] = "<SEP>".join(c for c in col.split("<SEP>") if c != key)

        return self._update(
            data=self.data[list(col_renames.keys())].rename(columns=col_renames)
        )

    def drop(self, *keys: str) -> "Mappings":
        assert len(set(self.data.columns)) == len(
            self.data.columns
        ), "Columns must be unique"

        if len(keys) != 1:
            for k in keys:
                self = self.drop(k)
            return self

        return self._update(data=self.data.drop(columns=self._get_cols(keys[0])))

    def sum(self, keep_key_index: list[int] | int | None = None) -> "Mappings":
        if len(self.data.columns) == 1:
            return self

        if isinstance(keep_key_index, int):
            keep_key_index = [keep_key_index]
        elif keep_key_index is None:
            keep_key_index = []

        columns = list(self.data.columns)
        for col in self.data.columns:
            if len(col.split("<SEP>")) != len(columns[0].split("<SEP>")):
                raise ValueError(
                    f"Can only sum columns with same-length keys. Try first calling "
                    f'access("key") or drop("key") to make all columns '
                    f"have the same number of keys."
                )

        if any(k < 0 or k >= len(columns[0].split("<SEP>")) for k in keep_key_index):
            raise ValueError(
                f"Keep indices must be in the range [0, {len(columns[0].split('<SEP>'))})"
            )

        target2sources = {}
        for col in columns:
            target = col.split("<SEP>")
            target = "<SEP>".join(target[i] for i in keep_key_index)
            target2sources.setdefault(target, []).append(col)

        new_data = pd.DataFrame(index=self.data.index)
        for target, sources in target2sources.items():
            new_data[target] = self.data[sources].sum(axis=1)

        return self._update(data=new_data)

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    def to_dict(self) -> dict[str, list[float]]:
        new = self.data.to_dict(orient="list")
        if len(self) == 1:
            new = {k: v[0] for k, v in new.items()}
        return new

    def per_compute(self, per_einsum: bool = False) -> "Mappings":
        new_df = self.data.copy()
        total_computes = self.num_computes()
        for col in new_df.columns:
            n_computes = total_computes
            if per_einsum:
                einsum_name = col.split("<SEP>")[0]
                if einsum_name not in self.einsum_names and einsum_name != "Total":
                    raise ValueError(
                        f"Einsum name {einsum_name} not found. Ensure that all "
                        f"columns are prefixed with the Einsum name if per_einsum "
                        f"is True."
                    )
                if einsum_name != "Total":
                    n_computes = self.num_computes(einsum_name)
            # Check if the column can be converted to numeric
            try:
                pd.to_numeric(new_df[col], errors="raise")
                new_df[col] /= n_computes
            except (ValueError, TypeError):
                # Skip columns that can't be converted to numeric
                continue
        return self._update(data=new_df)

    def drop_zeros(self) -> "Mappings":
        new_df = self.data.copy()
        new_df = new_df[(c for c in new_df.columns if (new_df[c] != 0).any())]
        return self._update(data=new_df)

    def render(self) -> str:
        if len(self) != 1:
            raise ValueError(
                f"Can only render a single mapping, but got {len(self)}. Try calling "
                f"mappings[i].render() instead, for some integer 0 <= i < {len(self)}."
            )
        return self.data.iloc[0][f"Total<SEP>mapping"].render()
