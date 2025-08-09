from pathlib import Path
import unittest

import islpy as isl

from fastfusion.frontend.mapping import Fill, Spatial, Temporal

class TestSimpleLinkTransferModel(unittest.TestCase):
    """
    Test the simple link transfer model. Adapted from the original test at the
    following link:
    https://github.com/NVlabs/timeloop/blob/master/src/unit-test/test-simple-link-transfer.cpp

    Code here:#include <boost/test/unit_test.hpp>

#include "isl-wrapper/ctx-manager.hpp"
#include "loop-analysis/spatial-analysis.hpp"

BOOST_AUTO_TEST_CASE(TestSimpleLinkTransferModel)
{
  using namespace analysis;

  auto fill = Fill(
    {Temporal(), Spatial(0, 0), Spatial(1, 0)},
    isl::map(
      GetIslCtx(),
      "{ [t, x, y] -> [t+x+y] : 0 <= x < 2 and 0 <= y < 2 and 0 <= t < 2}"
    )
  );

  auto occ = Occupancy(fill.dim_in_tags, fill.map);

  auto link_transfer_model = SimpleLinkTransferModel();

  auto info = link_transfer_model.Apply(0, fill, occ);

  BOOST_CHECK(info.fulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t = 1, x, y = 0] -> [1 + x] : 0 <= x <= 1; "
      "  [t = 1, x = 0, y] -> [1 + y] : 0 <= y <= 1 }"
    )
  ));

  BOOST_CHECK(info.unfulfilled_fill.map.is_equal(
    isl::map(
      GetIslCtx(),
      "{ [t = 0, x, y] -> [x + y] : 0 <= x <= 1 and 0 <= y <= 1; "
      "  [t = 1, x = 1, y = 1] -> [3] }"
    )
  ));
}
    """
    
    def test_simple_link_transfer_model(self):
        fill: Fill = Fill(
            {Temporal(), Spatial(0, 0), Spatial(1, 0)},
            isl.map(
                "{ [t, x, y] -> [t+x+y] : 0 <= x < 2 and 0 <= y < 2 and 0 <= t < 2}"
            )
        )
        # print(fill)
        

