from typing import Any, NamedTuple

from tqdm import tqdm
from fastfusion.accelerated_imports import pd
from fastfusion.frontend.workload.workload import EinsumName
from fastfusion.mapper.FFM._join_pmappings.sim import SIM
from fastfusion.mapper.FFM._pmapping_group.df_convention import COMPRESSED_INDEX, col_used_in_pareto, is_fused_loop_col, is_tensor_col
from fastfusion.mapper.FFM._pmapping_group.pmapping_group import PmappingGroup
from fastfusion.util.util import parallel, delayed


class DecompressData(NamedTuple):
    data: dict[EinsumName, dict[int, pd.DataFrame]]


def _compress(einsum_name: EinsumName, pmappings: SIM, start_index: int) -> tuple["PmappingGroup", pd.DataFrame]:
    data = pmappings.mappings.data
    data.reset_index(drop=True, inplace=True)
    data.index += start_index
    keep_cols = [c for c in data.columns if col_used_in_pareto(c) or is_fused_loop_col(c) or is_tensor_col(c)]
    compress_cols = [c for c in data.columns if c not in keep_cols]
    compressed_data = data[keep_cols].copy()
    decompress_data = data[compress_cols].copy()
    compressed_data[f"{einsum_name}<SEP>{COMPRESSED_INDEX}"] = data.index
    return PmappingGroup(compressed_data, skip_pareto=True), decompress_data


def _compress_pmapping_list(einsum_name: EinsumName, pmappings: list[SIM]) -> tuple[list[PmappingGroup], dict[int, pd.DataFrame]]:
    decompress_data = {}
    compressed = []
    start_index = 0
    jobs = []

    def job(start_index: int, pmappings: SIM):
        compress, decompress = _compress(
            einsum_name,
            pmappings,
            start_index,
        )
        compress = SIM(pmappings.compatibility, compress)
        compressed.append(compress)
        return compress, decompress, start_index

    for pmapping in pmappings:
        jobs.append(delayed(job)(start_index, pmapping))
        start_index += len(pmapping.mappings.data)

    for compress, decompress, start_index in parallel(jobs, n_jobs=1):
        compressed.append(compress)
        decompress_data[start_index] = decompress

    return compressed, decompress_data  # pd.concat(decompress_data)


def compress_einsum2pmappings(
    einsum2pmappings: dict[EinsumName, list[SIM]],
) -> tuple[dict[EinsumName, list[PmappingGroup]], DecompressData]:
    decompress_data = {}
    compressed_einsum2pmappings = {}

    jobs = []
    def job(einsum_name: EinsumName, pmappings: list[SIM]):
        compressed, decompress = _compress_pmapping_list(einsum_name, pmappings)
        return einsum_name, compressed, decompress

    for einsum_name, pmappings in einsum2pmappings.items():
        jobs.append(delayed(job)(einsum_name, pmappings))
        
    name_order = [einsum_name for einsum_name in einsum2pmappings.keys()]
    for einsum_name, compressed, decompress in parallel(jobs, pbar="Compressing pmappings", return_as="generator_unordered"):
        compressed_einsum2pmappings[einsum_name] = compressed
        decompress_data[einsum_name] = decompress
        
    compressed_einsum2pmappings = {einsum_name: compressed_einsum2pmappings[einsum_name] for einsum_name in name_order}
    decompress_data = {einsum_name: decompress_data[einsum_name] for einsum_name in name_order}

    # for einsum_name, pmappings in tqdm(einsum2pmappings.items(), desc="Compressing pmappings"):
    #     compressed, decompress = _compress_pmapping_list(einsum_name, pmappings)
    #     compressed_einsum2pmappings[einsum_name] = compressed
    #     decompress_data[einsum_name] = decompress
    return compressed_einsum2pmappings, DecompressData(decompress_data)


def decompress_pmappings(
    pmappings: PmappingGroup,
    decompress_data: DecompressData,
) -> PmappingGroup:
    data = pmappings.data
    for einsum_name, decompress in decompress_data.data.items():
        decompress_sub_dfs = []
        # We reverse it because we have the start index of each sub df, but we need the
        # end index so we know when to change to the next 
        decompressed_iter = reversed(decompress.items())
        start_index, chosen = float("inf"), None
        for i in reversed(sorted(set(data[f"{einsum_name}<SEP>{COMPRESSED_INDEX}"]))):
            while chosen is None or i < start_index:
                start_index, chosen = next(decompressed_iter)
            cur_chosen = chosen[chosen.index == i]
            assert len(cur_chosen) == 1, f"Expected 1 row, got {len(cur_chosen)}. Index {i}, start index {start_index}, chosen {list(chosen.index)}"
            decompress_sub_dfs.append(cur_chosen)
        data = pd.merge(
            data,
            pd.concat(decompress_sub_dfs),
            left_on=f"{einsum_name}<SEP>{COMPRESSED_INDEX}",
            right_index=True,
            how="left",
        )

    # Remove compressed_index columns that may have been created during
    # merge
    compressed_index_cols = [
        col for col in data.columns if COMPRESSED_INDEX in col
    ]
    if compressed_index_cols:
        data = data.drop(columns=compressed_index_cols)
    return PmappingGroup(data, skip_pareto=True)


# def _compress_data(
#         self,
#         einsum_name: EinsumName = None,
#         job_id: int = None,
#         skip_columns: list[str] = None,
#         keep_columns: list[str] = None,
#     ) -> pd.DataFrame:
#     self.data[MAPPING_INDEX_COLUMN] = self.data.index
#     keep_cols = [MAPPING_INDEX_COLUMN] + [c for c in self.data.columns if col_used_in_pareto(c) or c in skip_columns or c in keep_columns]
#     # recovery = self.data[[c for c in self.data.columns if c not in keep_cols] + [MAPPING_INDEX_COLUMN]]
#     # self._data = self.data[keep_cols]
#     recovery = self.data[[c for c in self.data.columns if c not in skip_columns]] # TODO: We may want to use the above if compressing uses too much memory
#     self._data = self.data[keep_cols].copy()
#     self._data[MAPPING_INDEX_COLUMN] = self._data[MAPPING_INDEX_COLUMN].apply(lambda x: (job_id, x))
#     if einsum_name is not None:
#         recovery.rename(columns={c: f"{einsum_name}{c}" for c in recovery.columns}, inplace=True)
#         self.data.rename(columns={MAPPING_INDEX_COLUMN: f"{einsum_name}{MAPPING_INDEX_COLUMN}"}, inplace=True)
#     return recovery

# def _decompress_data(self, einsum_name: EinsumName, decompress_data: list[DecompressData]):
#     src_idx_col = f"{einsum_name}{MAPPING_INDEX_COLUMN}"
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
# def compress_paretos(cls, einsum_name: EinsumName, paretos: list["PmappingGroup"], job_id: int, extra_data: dict[str, Any], skip_columns: list[str] = None, keep_columns: list[str] = None) -> DecompressData:
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
#     col = decompress_data.MAPPING_INDEX_COLUMN()
#     self.data[col] = self.data[col].apply(lambda x: (index, x))
