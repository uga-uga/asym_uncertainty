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
