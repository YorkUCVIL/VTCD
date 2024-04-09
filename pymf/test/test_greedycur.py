from pymf.greedycur import GREEDYCUR
import numpy as np


class TestGREEDYCUR:

    data = np.array([[0.25, 0.1, 0.0, 0.0], 
                     [1.0, 0.4, 0.7, 0.0],
                     [0.5, 0.125, 0.0, 0.1]])

    def test_compute_wh(self):
        mdl = GREEDYCUR(self.data, rrank=1, crank=1)

        mdl.factorize()
        
        # check if rows/columns are selected from data, everything else is a bit
        # difficult to test as a random selection is part of the CUR method.
        t = np.sum(self.data - mdl._C, axis=0)
        assert(0.0 in t)
        
        t = np.sum(self.data.T - mdl._R.T, axis=0)
        assert(0.0 in t)
