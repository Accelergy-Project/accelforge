from pathlib import Path
from unittest import TestCase

import accelforge as af

INPUT_FILES_DIR = Path(__file__).parent / "input_files" / "networked"


class TestParsing(TestCase):
    def test_hierarchical(self):
        spec = af.Spec.from_yaml(
            INPUT_FILES_DIR / "hierarchical.yaml",
        )
        self.assertIn("PeArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["PeArray"].get_fanout(), 1)
        self.assertIn("Scratchpad", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["Scratchpad"].get_fanout(), 4)
        self.assertIn("MacArray", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["MacArray"].get_fanout(), 1)

        try:
            spec = spec.calculate_component_costs()
        except af.EvaluationError as e:
            self.fail(e.message)

    def test_flat(self):
        spec = af.Spec.from_yaml(
            INPUT_FILES_DIR / "flat.yaml",
        )
        self.assertIn("NoC", spec.arch.nodes)
        self.assertEqual(spec.arch.nodes["NoC"].get_fanout(), 1)
        self.assertEqual(
            {n.name for n in spec.arch.get_nodes_of_type(af.spec.Leaf)},
            {
                "MainMemory",
                "NoC",
                "RowBuffer",
                "ColumnBuffer",
                "DistributedBuffer",
                "Scratchpad",
                "MAC",
            },
        )

        try:
            spec = spec.calculate_component_costs()
        except af.EvaluationError as e:
            self.fail(e.message)


class TestModel(TestCase):
    def test_hierarchical_1d(self):
        M = 8
        KN = 8
        MAC_TILE = 2
        M_TILE = 4
        BITS_PER_VALUE = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            # af.examples.arches.networked.hierarchical,
            INPUT_FILES_DIR / "hierarchical_1d.yaml",
            # af.examples.mappings.one_matmul_to_networked_hierarchical,
            INPUT_FILES_DIR / "one_matmul_to_networked_hierarchical_1d.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": 8,
                "KN": 8,
                "MAC_TILE": MAC_TILE,
                "M_TILE": M_TILE,
            },
        )
        result = spec.evaluate_mapping()
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE)  # number of used Scratchpad
            * M_TILE
            * KN  # temporal for n1 in mapping
            * sum(i for i in range(MAC_TILE))  # unicast along X-axis of MacArray
            * BITS_PER_VALUE,
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T1<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE)
            * M_TILE
            * KN  # temporal for n1 in mapping
            * (MAC_TILE - 1)   # multicast along X-axis of MacArray
            * BITS_PER_VALUE,
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>W0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE)
            * M_TILE
            * KN
            * sum(i for i in range(MAC_TILE))
            * BITS_PER_VALUE,
        )

        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * sum(i for i in range(KN // MAC_TILE))  # unicast along X-axis of PeArray
            * M_TILE
            * MAC_TILE
            * BITS_PER_VALUE,
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T1<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN // MAC_TILE - 1)  # multicast along X-axis of PeArray
            * M_TILE
            * KN
            * BITS_PER_VALUE,
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>W0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * sum(i for i in range(KN // MAC_TILE))  # unicast along PeArray
            * MAC_TILE
            * KN
            * BITS_PER_VALUE,
        )
        self.assertEqual(
            result.data["Total<SEP>latency"].iloc[0],
            4
        )

    def test_hierarchical(self):
        M = 8
        KN = 8
        MAC_TILE = 2
        PE_TILE = KN // MAC_TILE
        M_TILE = 4
        BITS_PER_VALUE = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            # af.examples.arches.networked.hierarchical,
            INPUT_FILES_DIR / "hierarchical.yaml",
            # af.examples.mappings.one_matmul_to_networked_hierarchical,
            INPUT_FILES_DIR / "one_matmul_to_networked_hierarchical.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": 8,
                "KN": 8,
                "MAC_TILE": MAC_TILE,
                "M_TILE": M_TILE,
            },
        )
        result = spec.evaluate_mapping()
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE) ** 2
            * M_TILE
            * (
                sum(i for i in range(MAC_TILE))  # unicasting along X
                +
                MAC_TILE * (MAC_TILE-1)  # multicast along Y for each column
            )
            * BITS_PER_VALUE,
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>T1<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE) ** 2
            * M_TILE
            * (
                MAC_TILE * (MAC_TILE - 1)  # multicast along X (the tile is shape N1, which is MAC_TILE here)
                +
                MAC_TILE * sum(i for i in range(MAC_TILE))  # unicasting along Y for each row
            )
            * BITS_PER_VALUE,
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>MacArray<SEP>W0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (KN / MAC_TILE) ** 2
            * M_TILE
            * (
                MAC_TILE * sum(i for i in range(MAC_TILE))  # unicast along X (the tile is shape N1, which is MAC_TILE here)
                +
                MAC_TILE * sum(i for i in range(MAC_TILE))  # unicasting along Y for each row
            )
            * BITS_PER_VALUE,
        )

        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (
                sum(i for i in range(PE_TILE))
                +
                PE_TILE * (PE_TILE - 1)
            )
            # tile shape
            * M_TILE * MAC_TILE * BITS_PER_VALUE,
        )
        # NOTE: assuming XY routing (as defined in mapping)
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>T1<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (
                PE_TILE * (PE_TILE - 1)
                +
                PE_TILE * sum(i for i in range(PE_TILE))
            )
            * M_TILE
            * MAC_TILE
            * BITS_PER_VALUE,
        )
        self.assertEqual(
            result.data["Matmul0<SEP>action<SEP>PeArray<SEP>W0<SEP>hops"].iloc[0],
            (M / M_TILE)
            * (
                PE_TILE * sum(i for i in range(PE_TILE))
                +
                PE_TILE * sum(i for i in range(PE_TILE))
            )
            * MAC_TILE**2
            * BITS_PER_VALUE,
        )

    def test_flat(self):
        M = 8
        KN = 8
        MAC_TILE = 2
        M_TILE = 4
        BITS_PER_VALUE = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            INPUT_FILES_DIR / "flat.yaml",
            INPUT_FILES_DIR / "one_matmul_to_flat.yaml",
            jinja_parse_data={
                "N_EINSUMS": 1,
                "M": 8,
                "KN": 8,
                "MAC_TILE": MAC_TILE,
                "M_TILE": M_TILE,
            },
        )
        result = spec.evaluate_mapping()
        self.assertEqual(
            result.data['Matmul0<SEP>action<SEP>NoC<SEP>T0<SEP>hops'].iloc[0],
            (
                M / M_TILE
                *
                (KN / MAC_TILE) * (KN / MAC_TILE - 1)   # num rows * multicast_hops
                *
                M_TILE * MAC_TILE  # tile shape
                *
                BITS_PER_VALUE
            )
        )
        self.assertEqual(
            result.data['Matmul0<SEP>action<SEP>NoC<SEP>T1<SEP>hops'].iloc[0],
            (
                M / M_TILE
                *
                (KN / MAC_TILE) * (KN / MAC_TILE - 1)   # num rows * multicast_hops
                *
                M_TILE * MAC_TILE  # tile shape
                *
                BITS_PER_VALUE
            )
        )
        self.assertEqual(
            result.data['Matmul0<SEP>action<SEP>NoC<SEP>W0<SEP>hops'].iloc[0],
            (
                M / M_TILE
                *
                (
                    4   # a 2x2 grid of physical buffers
                    *
                    (
                        sum(i for i in range(2)) * MAC_TILE  # unicast along row * tile shape
                        +
                        2 * sum(i for i in range(2))  # num cols * unicast down col
                    )
                )
                *
                MAC_TILE * MAC_TILE  # tile shape
                *
                BITS_PER_VALUE
            )
        )
        self.assertEqual(
            result.data['Matmul0<SEP>action<SEP>RowBuffer<SEP>T0<SEP>read'].iloc[0],
            (
                M / M_TILE
                *
                KN // MAC_TILE
                *
                M_TILE * MAC_TILE
                *
                BITS_PER_VALUE
            )
        )
        self.assertEqual(
            result.data['Matmul0<SEP>latency<SEP>RowBuffer'].iloc[0],
            (
                M / M_TILE
                *
                KN // MAC_TILE
                *
                M_TILE * MAC_TILE
                *
                BITS_PER_VALUE
                /
                4    # num of physical RowBuffer
            )
        )
        self.assertEqual(
            result.data['Matmul0<SEP>latency<SEP>DistributedBuffer'].iloc[0],
            (
                M / M_TILE
                *
                KN // MAC_TILE
                *
                KN // MAC_TILE
                *
                MAC_TILE * MAC_TILE  # tile shape
                *
                BITS_PER_VALUE
                /
                4    # num of physical DistributedBuffer
            )
        )


class TestMapper(TestCase):
    def test_hierarchical(self):
        M = 8
        KN = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            INPUT_FILES_DIR / "hierarchical.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN}
        )
        result = spec.map_workload_to_arch()

    def test_flat(self):
        M = 8
        KN = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            INPUT_FILES_DIR / "flat.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN}
        )
        result = spec.map_workload_to_arch()

    def test_flat_one_row_buffer(self):
        M = 8
        KN = 8

        spec = af.Spec.from_yaml(
            af.examples.workloads.matmuls,
            INPUT_FILES_DIR / "flat.yaml",
            jinja_parse_data={"N_EINSUMS": 1, "M": M, "KN": KN, "N_ROW_BUFFER": 1}
        )
        result = spec.map_workload_to_arch()
