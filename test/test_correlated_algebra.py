import pytest

from asym_uncertainty import Unc

class TestCorrelatedAlgebra(object):
    def test_correlated_algebra(self):
        # Test the algebra with correlated quantities, i.e. calculations like
        # a*a, a+a, a-a ...
        # where the probability distributions for the operands can not be assumed
        # to be independent

        a = Unc(1., 1., 1.)

        add = a+a
        assert add.mean_value == 2.
        assert add.sigma_low == 2.
        assert add.sigma_up == 2.

        sub = a-a
        assert sub.mean_value == 0.
        assert sub.sigma_low == 0.
        assert sub.sigma_up == 0.

        ratio = a/a 
        assert ratio.mean_value == 1.
        assert ratio.sigma_low == 0.
        assert ratio.sigma_up == 0.
