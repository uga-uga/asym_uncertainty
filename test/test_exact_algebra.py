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

from asym_uncertainty import Unc

class TestExactAlgebra(object):
    def test_exact_Unc(self):
        a = Unc(1., 0., 0.)
        b = Unc(1., 0., 0.)

        add = a + b
        sub = a - b
        mult = a*b
        ratio = a/b
        power = a**b

        assert add.mean_value == 2.
        assert add.sigma_low == 0.
        assert add.sigma_up == 0.

        assert sub.mean_value == 0.
        assert sub.sigma_low == 0.
        assert sub.sigma_up == 0.

        assert mult.mean_value == 1.
        assert mult.sigma_low == 0.
        assert mult.sigma_up == 0.

        assert ratio.mean_value == 1.
        assert ratio.sigma_low== 0.
        assert ratio.sigma_low == 0.

        assert power.mean_value == 1.
        assert power.sigma_low== 0.
        assert power.sigma_low == 0.

        a = Unc(1., 1., 1.)
        b = Unc(1., 0., 0.)

        add = a + b
        sub = a - b
        mult = a*b
        ratio = a/b
        # The power operator __pow__() a**b is already tested in test_algebra.py

        assert add.mean_value == 2.
        assert add.sigma_low == 1.
        assert add.sigma_up == 1.

        assert sub.mean_value == 0.
        assert sub.sigma_low == 1.
        assert sub.sigma_up == 1.

        assert mult.mean_value == 1.
        assert mult.sigma_low == 1.
        assert mult.sigma_up == 1.

        assert ratio.mean_value == 1.
        assert ratio.sigma_low== 1.
        assert ratio.sigma_low == 1.

    def test_float_vs_Unc(self):
        # Test __radd__() and __rsub__() which determines the addition/subtraction of a normal float and Unc
        a = Unc(1., 0.1, 0.1)

        add = 1. + a
        sub = 1. - a

        assert add.mean_value == 2.
        assert add.sigma_low == 0.1
        assert add.sigma_up == 0.1

        assert sub.mean_value == 0.
        assert sub.sigma_low == 0.1
        assert sub.sigma_up == 0.1

        add = a + 1.
        sub = a - 1.

        assert add.mean_value == 2.
        assert add.sigma_low == 0.1
        assert add.sigma_up == 0.1

        assert sub.mean_value == 0.
        assert sub.sigma_low == 0.1
        assert sub.sigma_up == 0.1

        # Test __rmul__()  which determines the multiplication of a normal float and Unc
        a = Unc(1., 1., 1.)
        mult = 1.*a

        assert mult.mean_value == 1.
        assert mult.sigma_low == 1.
        assert mult.sigma_up == 1.

        mult = a*1.

        assert mult.mean_value == 1.
        assert mult.sigma_low == 1.
        assert mult.sigma_up == 1.

        # Test __rtruediv__() which determine Unc/float or Unc/int
        ratio = a/1.

        assert ratio.mean_value == 1.
        assert ratio.sigma_low== 1.
        assert ratio.sigma_up == 1.
