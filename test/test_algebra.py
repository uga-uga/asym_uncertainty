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

SQRT2 = 1.4142135623730951
STATISTICAL_UNCERTAINTY_LIMIT = 0.15 # Maximum tolerated relative deviation from exact result
MULTIPLICATION_SIGMA = 0.68327
DIVISION_SIGMA = 1.8374
INVERSE_MEAN = 0.5 

class TestUncAlgebra(object):
    def test_addition(self):
        a = Unc(1., 0.1, 0.1)
        b = Unc(1., 0.1, 0.1)

        add = a + b

        # For addition ( __add__() ), the expected uncertainty of the result is sqrt(sigma_1**2 + sigma_2**2)
        assert add.mean_value >= 2. - STATISTICAL_UNCERTAINTY_LIMIT*2. and add.mean_value <= 2. + STATISTICAL_UNCERTAINTY_LIMIT*2.
        assert add.sigma_low >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and add.sigma_low <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1
        assert add.sigma_up >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and add.sigma_up <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1

    def test_subtraction(self):
        a = Unc(1., 0.1, 0.1)
        b = Unc(1., 0.1, 0.1)

        sub = a - b

        # For subtraction ( __sub__() ), the expected uncertainty of the result is sqrt(sigma_1**2 + sigma_2**2)
        assert sub.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and sub.mean_value <= 2. + STATISTICAL_UNCERTAINTY_LIMIT*2.
        assert sub.sigma_low >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and sub.sigma_low <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1
        assert sub.sigma_up >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and sub.sigma_up <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1

    def test_multiplication(self):
        a = Unc(0., 1., 1.)
        b = Unc(0., 1., 1.)

        mult = a*b

        # For multiplication ( __mul__() ) of two normal distributed quantities with mean value == 0 and sigma == 1, the expected product distribution is K_0(|x|)/pi, where K_0 is the modified Bessel function of the second kind of order 0
        # By numerical integration, one finds that the 68.27% coverage interval is about [-0.68327, 0.68327]
        assert mult.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and mult.mean_value <= 0. + STATISTICAL_UNCERTAINTY_LIMIT*2.
        assert mult.sigma_low >= MULTIPLICATION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA and mult.sigma_low <= MULTIPLICATION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA
        assert mult.sigma_up >= MULTIPLICATION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA and mult.sigma_up <= MULTIPLICATION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA

    def test_division(self):
        a = Unc(0., 1., 1.)
        b = Unc(0., 1., 1.)

        ratio = a/b
        # For the ratio ( __truediv__() ) of two normal distributed quantities with mean value == 0 and sigma == 1, the expected ratio distribution is a Cauchy distribution.
        # By numerical integration, one finds that the 68.27% coverage interval is about [-1.8374, 1.8374]

        assert ratio.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and ratio.mean_value <= 0. + STATISTICAL_UNCERTAINTY_LIMIT*2.
        assert ratio.sigma_low >= DIVISION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*DIVISION_SIGMA and ratio.sigma_low <= DIVISION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*DIVISION_SIGMA
        assert ratio.sigma_up >= DIVISION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*DIVISION_SIGMA and ratio.sigma_up <= DIVISION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*DIVISION_SIGMA

        # Test __rtruediv__()  which determines the division of a normal float and Unc, by calculating the inverse of a random variable
        # For this, take the expected ratio distribution for the division of two normal distributed random variables with different mean value and sigma (D.V. Hinkley, Biometrika 56, 3 (1969) 635).
        # By numerical integration (choose sigma1 << sigma2 to simulate a discrete number divided by one with uncertainty), one finds that the most probable value of the resulting distribution is 0.5
        a = Unc(1., 1., 1.)
        ratio = 1./a

        assert ratio.mean_value >= INVERSE_MEAN - STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN and ratio.mean_value <= INVERSE_MEAN + STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN 

        ratio = Unc(1., 0., 0.)/a

        assert ratio.mean_value >= INVERSE_MEAN - STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN and ratio.mean_value <= INVERSE_MEAN + STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN 

    def test_power(self):
        # Testing the exponential x^y is more difficult, because the resulting distributions are
        # different if x, y or both are values with uncertainties.

        # For the case where only y has a normal distribution and x is exact, the formula can
        # be rewritten as 
        # x**y = exp(y*ln(x))
        # In a more general formulation
        # z = exp(mu + sigma*y)
        # where y is the normal-distributed variable, the result z has a log-normal distribution
        # with mean value m and variance v, which are related to mu and sigma via
        # mu = ln(m/sqrt(1 + v**2/(m**2))) and sigma = sqrt(ln(1 + v**2/(m**2)))
        # In the example, mu = 0 and sigma = ln(x) with x = 2
        # Then, the mean value of the log-normal distribution should be about 1.27
        a = Unc(1., 1., 1.)

        power = 2.**a

        assert power.mean_value >= 1.27 - STATISTICAL_UNCERTAINTY_LIMIT*1.27 and power.mean_value <= 1.27 + STATISTICAL_UNCERTAINTY_LIMIT*1.27

        # The same with an exact Unc number:
        b = Unc(2., 0., 0.)

        power = b**a

        assert power.mean_value >= 1.27 - STATISTICAL_UNCERTAINTY_LIMIT*1.27 and power.mean_value <= 1.27 + STATISTICAL_UNCERTAINTY_LIMIT*1.27

        # For the case where y is exact and x has a normal distribution, no general expression
        # exists. However, x**1 should yield x.

        power = a**1.

        assert power.mean_value >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.mean_value <= 1. + STATISTICAL_UNCERTAINTY_LIMIT
        assert power.sigma_low >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.sigma_low <= 1. + STATISTICAL_UNCERTAINTY_LIMIT
        assert power.sigma_up >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.sigma_up <= 1. + STATISTICAL_UNCERTAINTY_LIMIT

        # Of course, the case where both values are normally distributed also does not have
        # an analytical expression, since it is a combination of the previous case.
        # Compared to the previous case, use a number with uncertainty for y, but keep its
        # uncertainty small, so that at least this part of the code is executed.

        a = Unc(1., 0.01, 0.01)
        b = Unc(1., 0.01, 0.01)

        power = a**b

        assert power.mean_value >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.mean_value <= 1. + STATISTICAL_UNCERTAINTY_LIMIT
        assert power.sigma_low >= 0.01 - STATISTICAL_UNCERTAINTY_LIMIT*0.01 and power.sigma_low <= 0.01 + STATISTICAL_UNCERTAINTY_LIMIT*0.01
        assert power.sigma_up >= 0.01 - STATISTICAL_UNCERTAINTY_LIMIT*0.01 and power.sigma_up <= 0.01 + STATISTICAL_UNCERTAINTY_LIMIT*0.01
