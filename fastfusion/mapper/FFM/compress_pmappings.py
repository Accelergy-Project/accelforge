from typing import Any, NamedTuple
from fastfusion.accelerated_imports import pd
from fastfusion.frontend.workload.workload import EinsumName


COMPRESSED_INDEX_COLUMN = "__COMPRESSED_INDEX"


class DecompressData(NamedTuple):
    einsum_name: EinsumName
    data: pd.DataFrame
    extra_data: dict

    def compressed_index_column(self):
        return f"{self.einsum_name}{COMPRESSED_INDEX_COLUMN}"


class GroupedDecompressData(NamedTuple):
    prefix2datalist: dict[EinsumName, dict[int, DecompressData]]

    def register_decompress_data(
        self,
        einsum_name: EinsumName,
        job_ids: list[int],
        decompress_data: DecompressData,
    ):
        target = self.prefix2datalist.setdefault(einsum_name, {})
        for i, job_id in enumerate(job_ids):
            assert job_id not in target
            target[job_id] = decompress_data
            # if i == 0:
            #     target[job_id] = decompress_data
            # # Store a reference to the first decompress data so it's not copied a bunch
            # # of times when we send this to subprocesses.
            # else:
            #     target[job_id] = target[job_ids[0]]

    def get_decompress_data(
        self, einsum_name: EinsumName, job_id: int
    ) -> DecompressData:
        einsum_data = self.prefix2datalist[einsum_name]
        found = einsum_data[job_id]
        if not isinstance(found, DecompressData):
            found = einsum_data[found]
        return found


def compress_df(
    df: pd.DataFrame,
    einsum_name: EinsumName,
    compress_columns: list[str],
    keep_columns: list[str],
    extra_data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert (
        COMPRESSED_INDEX_COLUMN in df.columns
    ), f"Expected {COMPRESSED_INDEX_COLUMN} in df"
    df[COMPRESSED_INDEX_COLUMN] = df.apply(
        lambda row: (int(row[COMPRESSED_INDEX_COLUMN]), row.name),
        axis=1,
    )
    if COMPRESSED_INDEX_COLUMN not in compress_columns:
        compress_columns = [COMPRESSED_INDEX_COLUMN] + compress_columns
    if COMPRESSED_INDEX_COLUMN not in keep_columns:
        keep_columns = [COMPRESSED_INDEX_COLUMN] + keep_columns
    recovery = df[compress_columns]
    df = df[keep_columns]
    if einsum_name is not None:
        recovery = recovery.rename(
            columns={c: f"{einsum_name}{c}" for c in recovery.columns}
        )
        df = df.rename(
            columns={COMPRESSED_INDEX_COLUMN: f"{einsum_name}{COMPRESSED_INDEX_COLUMN}"}
        )

    return df, DecompressData(einsum_name, recovery, extra_data)


def decompress_joined_df(df: pd.DataFrame, decompress_data: GroupedDecompressData):
    for einsum_name, decompress_data_list in decompress_data.prefix2datalist.items():
        src_idx_col = f"{einsum_name}{COMPRESSED_INDEX_COLUMN}"
        rows = []
        extra_data = []
        for r in df[src_idx_col]:
            assert isinstance(
                r, tuple
            ), f"Expected tuple, got {type(r)}. Was register_decompress_data called with this Pareto?"
            assert len(r) == 2, f"Expected tuple of length 2, got {r}"
            extra_data_index, decompress_data_list_index = r
            data = decompress_data_list[extra_data_index]
            # Find the row in the data with src_idx_col == row_index
            row = data.data[data.data[src_idx_col] == r]
            assert len(row) == 1, f"Expected 1 row, got {len(row)}"
            rows.append(row)
            extra_data.append(data.extra_data[extra_data_index])

        new_df = pd.concat(rows)
        extra_data_df = pd.DataFrame(extra_data)

        # Drop the src_idx_col
        df.drop(columns=[src_idx_col], inplace=True)
        # Reset index to avoid duplicate index values that cause InvalidIndexError
        a = new_df.reset_index(drop=True)
        b = extra_data_df.reset_index(drop=True)
        c = df.reset_index(drop=True)
        df = pd.concat([a, b, c], axis=1)
        df.drop(columns=[src_idx_col], inplace=True)
    return df


# def _compress_data(
#         self,
#         einsum_name: EinsumName = None,
#         job_id: int = None,
#         skip_columns: list[str] = None,
#         keep_columns: list[str] = None,
#     ) -> pd.DataFrame:
#     self.data[COMPRESSED_INDEX_COLUMN] = self.data.index
#     keep_cols = [COMPRESSED_INDEX_COLUMN] + [c for c in self.data.columns if col_used_in_pareto(c) or c in skip_columns or c in keep_columns]
#     # recovery = self.data[[c for c in self.data.columns if c not in keep_cols] + [COMPRESSED_INDEX_COLUMN]]
#     # self._data = self.data[keep_cols]
#     recovery = self.data[[c for c in self.data.columns if c not in skip_columns]] # TODO: We may want to use the above if compressing uses too much memory
#     self._data = self.data[keep_cols].copy()
#     self._data[COMPRESSED_INDEX_COLUMN] = self._data[COMPRESSED_INDEX_COLUMN].apply(lambda x: (job_id, x))
#     if einsum_name is not None:
#         recovery.rename(columns={c: f"{einsum_name}{c}" for c in recovery.columns}, inplace=True)
#         self.data.rename(columns={COMPRESSED_INDEX_COLUMN: f"{einsum_name}{COMPRESSED_INDEX_COLUMN}"}, inplace=True)
#     return recovery

# def _decompress_data(self, einsum_name: EinsumName, decompress_data: list[DecompressData]):
#     src_idx_col = f"{einsum_name}{COMPRESSED_INDEX_COLUMN}"
#     rows = []
#     extra_data = []
#     for r in self.data[src_idx_col]:
#         assert isinstance(r, tuple), \
#             f"Expected tuple, got {type(r)}. Was register_decompress_data called with this Pareto?"
#         assert len(r) == 2, f"Expected tuple of length 2, got {r}"
#         data_index, row_index = r
#         data = decompress_data[data_index]
#         # Find the row in the data with src_idx_col == row_index
#         row = data.data[data.data[src_idx_col] == row_index]
#         assert len(row) == 1, f"Expected 1 row, got {len(row)}"
#         rows.append(row)
#         extra_data.append(data.extra_data)

#     new_df = pd.concat(rows)
#     extra_data_df = pd.DataFrame(extra_data)

#     # Drop the src_idx_col
#     self.data.drop(columns=[src_idx_col], inplace=True)
#     # Reset index to avoid duplicate index values that cause InvalidIndexError
#     a = new_df.reset_index(drop=True)
#     b = extra_data_df.reset_index(drop=True)
#     c = self.data.reset_index(drop=True)
#     self._data = pd.concat([a, b, c], axis=1)
#     self._data.drop(columns=[src_idx_col], inplace=True)

# def decompress(self, decompress_data: GroupedDecompressData):
#     for einsum_name, decompress_data_list in decompress_data.prefix2datalist.items():
#         assert decompress_data_list, f"No decompress data found for {einsum_name}"
#         self._decompress_data(einsum_name, decompress_data_list)


# @classmethod
# def compress_paretos(cls, einsum_name: EinsumName, paretos: list["PartialMappings"], job_id: int, extra_data: dict[str, Any], skip_columns: list[str] = None, keep_columns: list[str] = None) -> DecompressData:
#     index = 0
#     decompress_data = []
#     for p in paretos:
#         p.data.reset_index(drop=True, inplace=True)
#         p.data.index += index
#         index += len(p.data)
#         decompress_data.append(p._compress_data(einsum_name, job_id, skip_columns, keep_columns))

#     decompress_data = pd.concat(decompress_data) if decompress_data else pd.DataFrame()
#     decompress_data.reset_index(drop=True, inplace=True)
#     return DecompressData(
#         einsum_name=einsum_name,
#         decompress_data=decompress_data,
#         extra_data={f"{einsum_name}{k}": v for k, v in extra_data.items()},
#     )

# def _tuplefy_compress_data(self, index: int, decompress_data: DecompressData):
#     col = decompress_data.compressed_index_column()
#     self.data[col] = self.data[col].apply(lambda x: (index, x))
