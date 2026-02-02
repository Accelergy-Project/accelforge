from pathlib import Path
import unittest

from accelforge.frontend import Workload
from accelforge.frontend.workload import *


class TestWorkload(unittest.TestCase):
    def test_mha(self):
        workload = Workload.from_yaml(Path(__file__).parent / "mha.yaml")

        TENSOR_NAMES_REF = {
            "I",
            "WV",
            "V",
            "WK",
            "K",
            "WQ",
            "Q",
            "QK",
            "AV",
            "WZ",
            "Z",
            "WFFA",
            "FFA",
            "WFFB",
            "FFB",
        }
        self.assertEqual(TENSOR_NAMES_REF, {t for t in workload.tensor_names})

        rank_variable_bounds = get_rank_variable_bounds(workload, "Q")
        REF_RANK_VARIABLE_BOUNDS = {"m": 2, "b": 2, "h": 2, "e": 2, "d": 4}
        self.assertEqual(REF_RANK_VARIABLE_BOUNDS, rank_variable_bounds)

        REF_TENSOR_Q_SIZE = 16  # B*H*M*E
        self.assertEqual(REF_TENSOR_Q_SIZE, get_tensor_size(workload, TensorName("Q")))

    def test_conv(self):
        workload = Workload.from_yaml(Path(__file__).parent / "mobilenet.workload.yaml")
        self.assertEqual(
            workload.get_pairwise_equivalent_rank_variables(),
            {
                "pb0": {"pa0", "pb0"},
                "qb0": {"qa0", "qb0"},
                "t0": {"t0"},
                "pa0": {"r0", "pb0"},
                "qa0": {"qb0", "s0"},
                "r0": {"pa0"},
                "s0": {"qa0"},
            },
        )

    def test_conv_initial_deltas(self):
        workload = Workload.from_yaml(
            Path(__file__).parent / "mobilenet_long.workload.yaml"
        )

        initial_deltas = get_stride_and_halo(workload)
        self.assertEqual(
            {
                ("PwiseA0", "WA0"): {("N0", "n0"): (1, 0), ("T0", "t0"): (1, 0)},
                ("PwiseA0", "T0"): {
                    ("P0", "p0"): (1, 0),
                    ("Q0", "q0"): (1, 0),
                    ("N0", "n0"): (1, 0),
                },
                ("PwiseA0", "TA0"): {
                    ("P0", "p0"): (1, 0),
                    ("Q0", "q0"): (1, 0),
                    ("T0", "t0"): (1, 0),
                },
                ("Dwise0", "WAB0"): {
                    ("R0", "r0"): (1, 0),
                    ("S0", "s0"): (1, 0),
                    ("T0", "t0"): (1, 0),
                },
                ("Dwise0", "TB0"): {
                    ("P1", "p1"): (1, 0),
                    ("Q1", "q1"): (1, 0),
                    ("T0", "t0"): (1, 0),
                },
                ("Dwise0", "TA0"): {
                    ("P0", "r0"): (1, 6),
                    ("P0", "p1"): (1, 2),
                    ("Q0", "q1"): (1, 2),
                    ("Q0", "s0"): (1, 6),
                    ("T0", "t0"): (1, 0),
                },
                ("PwiseB0", "WB0"): {("T0", "t0"): (1, 0), ("N1", "n1"): (1, 0)},
                ("PwiseB0", "T1"): {
                    ("P1", "p1"): (1, 0),
                    ("Q1", "q1"): (1, 0),
                    ("N1", "n1"): (1, 0),
                },
                ("PwiseB0", "TB0"): {
                    ("P1", "p1"): (1, 0),
                    ("Q1", "q1"): (1, 0),
                    ("T0", "t0"): (1, 0),
                },
                ("PwiseA1", "TA1"): {
                    ("P1", "p1"): (1, 0),
                    ("Q1", "q1"): (1, 0),
                    ("T1", "t1"): (1, 0),
                },
                ("PwiseA1", "T1"): {
                    ("P1", "p1"): (1, 0),
                    ("Q1", "q1"): (1, 0),
                    ("N1", "n1"): (1, 0),
                },
                ("PwiseA1", "WA1"): {("N1", "n1"): (1, 0), ("T1", "t1"): (1, 0)},
                ("Dwise1", "TA1"): {
                    ("P1", "p2"): (1, 2),
                    ("P1", "r1"): (1, 6),
                    ("Q1", "q2"): (1, 2),
                    ("Q1", "s1"): (1, 6),
                    ("T1", "t1"): (1, 0),
                },
                ("Dwise1", "TB1"): {
                    ("P2", "p2"): (1, 0),
                    ("Q2", "q2"): (1, 0),
                    ("T1", "t1"): (1, 0),
                },
                ("Dwise1", "WAB1"): {
                    ("R1", "r1"): (1, 0),
                    ("S1", "s1"): (1, 0),
                    ("T1", "t1"): (1, 0),
                },
                ("PwiseB2", "T2"): {
                    ("P2", "p2"): (1, 0),
                    ("Q2", "q2"): (1, 0),
                    ("N2", "n2"): (1, 0),
                },
                ("PwiseB2", "WB1"): {("T1", "t1"): (1, 0), ("N2", "n2"): (1, 0)},
                ("PwiseB2", "TB1"): {
                    ("P2", "p2"): (1, 0),
                    ("Q2", "q2"): (1, 0),
                    ("T1", "t1"): (1, 0),
                },
            },
            initial_deltas,
        )
