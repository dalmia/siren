import unittest
import math
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np
from siren.init import siren_uniform_


class SineUniformCase(unittest.TestCase):
    """Class to test the siren weight intialization function"""
    def test_siren_uniform_(self):
        w = torch.empty(3, 5)
        siren_uniform_(w, mode='fan_in', c=6)
        self.assertEqual(torch.abs(w).gt(math.sqrt(2)).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
