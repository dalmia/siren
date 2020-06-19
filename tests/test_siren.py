import unittest
from numpy.testing import assert_array_almost_equal
import torch
import numpy as np
from siren.siren import Sine, SIREN


class SineTestCase(unittest.TestCase):
    """Class to test the Sine activation function"""
    def test_sine(self):
        dummy = torch.FloatTensor([np.pi, np.pi / 2])
        sine = Sine(w0=1)
        out = sine(dummy)
        target = torch.FloatTensor([0, 1])
        assert_array_almost_equal(target, out, decimal=6)


class SIRENTestCase(unittest.TestCase):
    """Class to test the SIREN model"""
    def test_siren(self):
        in_features = 10
        layers = [64, 128]
        out_features = 5
        dummy = torch.ones(in_features)
        model = SIREN(layers, in_features, out_features)
        out = model(dummy)

        self.assertEqual(out.shape, (5,))


if __name__ == "__main__":
    unittest.main()
