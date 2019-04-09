#    This file is part of asym_uncertainty.
#
#    asym_uncertainty is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    asym_uncertainty is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with asym_uncertainty.  If not, see <http://www.gnu.org/licenses/>.

import pytest

from numpy import mean, std

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

        # Check setting new value for n_random
        a = Unc(1., 0.5, 0.5)
        assert a.n_random == int(1e6)
        with pytest.raises(ValueError):
            a = Unc(1., 0.5, 0.5, n_random = 2.5)
        with pytest.raises(ValueError):
            a = Unc(1., 0.5, 0.5, n_random = -1)
        a = Unc(1., 0.5, 0.5, n_random = 100)
        assert a.n_random == 100
        a.set_n_random(1000)
        assert a.n_random == 1000

        # Check that, if the value of n_random is changed, the size of the random_values
        # array is adjusted accordingly if store = True
        a = Unc(1., 0.5, 0.5, n_random = 100, store=True)
        assert len(a.random_values) == 100
        a.set_n_random(50)
        assert len(a.random_values) == 50
        a.set_n_random(100)
        assert len(a.random_values) == 100

    def test_Unc_output(self):
        b = Unc(1., 0., 0.)

        assert b.__repr__() == "Unc( mean_value=1.0, sigma_low=0.0, sigma_up=0.0, limits=[-inf, inf] )"
        b.set_limits([-1., 2.])
        assert b.__repr__() == "Unc( mean_value=1.0, sigma_low=0.0, sigma_up=0.0, limits=[-1.0, 2.0] )"
        b.store=True
        assert b.__repr__() == "Unc( mean_value=1.0, sigma_low=0.0, sigma_up=0.0, limits=[-1.0, 2.0], store=True )"
        assert b.__str__() == "1.0 - 0.0 + 0.0"

    def test_random_sampling(self):
        a = Unc(0., 1., 1., store=True)

        mean_value = mean(a.random_values)
        sigma = std(a.random_values)

        assert -0.1 <= mean_value <= 0.1
        assert 0.95 <= sigma <= 1.05

        b = Unc(0., 1., 1.)

        assert len(b.random_values) == 1
        assert b.random_values[0] == 0.
