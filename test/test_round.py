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

class TestRound(object):
    def test_round(self):
        # Test with the examples of the PDG

        a = Unc(0.827, 0.119, 0.119)

        assert a.rounded[0] == 0.83 
        assert a.rounded[1] == 0.12
        assert a.rounded[2] == 0.12 

        a = Unc(0.827, 0.367, 0.367)

        assert a.rounded[0] == 0.8
        assert a.rounded[1] == 0.4
        assert a.rounded[2] == 0.4 

        # Test the case when the uncertainty is asymmetric.
        # Then, the lower of the two uncertainties determines the digits of
        # the rounded result
        a = Unc(0.827, 0.119, 0.367)

        assert a.rounded[0] == 0.83 
        assert a.rounded[1] == 0.12
        assert a.rounded[2] == 0.37

        # Test the case that the mean value is the quantity with the
        # lowest value
        a = Unc(0.827, 0.960, 0.970)

        assert a.rounded[0] == 0.8
        assert a.rounded[1] == 1.
        assert a.rounded[2] == 1.

        # Test the case when mean_value << sigma_low, sigma_up
        a = Unc(0.00827, 0.960, 0.970)

        assert a.rounded[0] == 0.
        assert a.rounded[1] == 0.96
        assert a.rounded[2] == 0.97

        # Test the case mean_value == sigma_low == sigma_up == 0.
        a = Unc(0., 0., 0.)

        assert a.rounded[0] == 0.
        assert a.rounded[1] == 0.
        assert a.rounded[2] == 0.
        
        # Check the updates of the rounded values when the
        # 'set_*' methods are called
        a.set_sigma_low(0.123)
        a.set_sigma_up(0.321)
        a.set_mean_value(0.0456)

        assert a.rounded[0] == 0.05
        assert a.rounded[1] == 0.12
        assert a.rounded[2] == 0.32

        # Check string representation
        assert a.__repr__() == "Unc( mean_value=0.05, sigma_low=0.12, sigma_up=0.32, limits=[-inf, inf] )"
