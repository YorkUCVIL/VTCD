from pymf.bnmf import BNMF
from base import assert_set_equal
import numpy as np


class TestCNMF():

    data = np.array([[0, 1.0, 1.0], 
                     [1.0, 0, 1.0]])

    W = np.array([[0.0, 1.0], 
                  [1.0, 0.0]])

    H = np.array([[1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]])

    def test_cnmf(self):
        mdl = BNMF(self.data, num_bases=2)

        mdl.factorize(niter=100)
        assert_set_equal(mdl.W, self.W, decimal=1)

        assert_set_equal(mdl.H.T, self.H, decimal=1)
        rec = mdl.frobenius_norm()
        assert(rec <= 0.1)



