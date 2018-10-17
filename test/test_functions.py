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
