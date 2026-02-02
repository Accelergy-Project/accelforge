"""
Tests some of the key custom isl functions.
"""

import unittest

import islpy as isl

from accelforge.model._looptree.reuse.isl.isl_functions import *

class BasicIslFunctionTests(unittest.TestCase):
    """
    Prima facie checks of `isl_functions` to ensure basic functionality.
    """

    def test_project_dim_after(self):
        pass