import pytest

from asym_uncertainty import exp, Unc

class TestIO(object):
    def test_Unc_input(self):

        a = Unc(1., 0., 0.)
        b = Unc(1., 1., 0.)
        assert a.is_exact
        assert not b.is_exact

        with pytest.raises(ValueError):
            Unc(1., -1., 1.)
        with pytest.raises(ValueError):
            Unc(1., 1., -1.)
        with pytest.raises(ValueError):
            a.set_sigma_low(-1.)
        with pytest.raises(ValueError):
            a.set_sigma_up(-1.)

        b.set_sigma_low(0.)
        assert b.is_exact

        b.set_mean_value(-1.)
        assert b.mean_value == -1.

        b.set_sigma_up(1.)
        assert b.sigma_up== 1.

        a = 'a'
        with pytest.raises(ValueError):
            c = a+b
        with pytest.raises(ValueError):
            c = b+a
        with pytest.raises(ValueError):
            c = a-b
        with pytest.raises(ValueError):
            c = b-a
        with pytest.raises(ValueError):
            c = a*b
        with pytest.raises(ValueError):
            c = b*a
        with pytest.raises(ValueError):
            c = a/b
        with pytest.raises(ValueError):
            c = b/a
        with pytest.raises(ValueError):
            c = exp(a)

        # Unary '-' operator
        a = Unc(1., 0., 0.)
        b = -a

        assert b.mean_value == -1.

        with pytest.raises(ValueError):
            a = Unc(1., 1., 1., limits=[1., 0.])
        with pytest.raises(ValueError):
            a = Unc(1., 1., 1., limits=[1., 0., 2.])
        a = Unc(1., 1., 1., limits=[0., 1.])
        with pytest.raises(ValueError):
            a.set_lower_limit(2.)
        with pytest.raises(ValueError):
            a.set_upper_limit(-1.)
        with pytest.raises(ValueError):
            a.set_limits([1., 0.])

        a.set_lower_limit(0.)
        a.set_upper_limit(1.)
        assert a.limits[0] == 0.
        assert a.limits[1] == 1.
        # Try setting new limits which are outside the old ones. This should cause a
        # RuntimeWarning
        with pytest.raises(RuntimeWarning):
            a.set_limits([3., 4.])

        a.set_limits([0.1, 0.9])
        assert a.limits[0] == 0.1
        assert a.limits[1] == 0.9

        # Check that the numerical values of mean_value, sigma_low and sigma_up are reset
        # when the limits are changed

        a = Unc(0., 1., 1.)
        a.set_limits([-1., 1.])

        # A normal distribution restricted to the range [-1, 1] should have its 68.27 % coverage
        # interval between ~[-0.622, 0.622]
        assert -0.622 <= a.mean_value <= 0.622
        assert a.mean_value - a.sigma_low > -1.
        assert a.mean_value + a.sigma_up < 1.

    def test_Unc_output(self):
        b = Unc(1., 0., 0.)

        assert b.__repr__() == "1.0 - 0.0 + 0.0"
        assert b.__str__() == "1.0_{0.0}^{0.0}"
