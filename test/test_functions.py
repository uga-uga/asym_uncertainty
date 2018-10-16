import pytest

from numpy import exp as nexp

from asym_uncertainty import exp, Unc

class TestFunctions(object):
    def test_exp(self):
        # Test exp by showing that exp(Unc(1., sigma, sigma)) gets closer to 
        # e if sigma is reduced

        a = Unc(1., 1., 1.)
        c = exp(a)
        assert c.mean_value < 1.2

        a = Unc(1., 0.5, 0.5)
        c = exp(a)
        assert c.mean_value < 2.3

        a = Unc(1., 0.05, 0.05)
        c = exp(a)
        assert c.mean_value >= 2.65 and c.mean_value <= 2.8

        a = Unc(1., 0., 0.)
        c = exp(a)
        assert c.mean_value == nexp(1.)
