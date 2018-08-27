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

from math import inf

import pytest
from numpy import array, concatenate, ones
from numpy import exp as nexp

from asym_uncertainty import exp, Unc

STATISTICAL_UNCERTAINTY_LIMIT = 0.25 # Maximum tolerated relative deviation from exact result
SQRT2 = 1.4142135623730951
MULTIPLICATION_SIGMA = 0.68327
RATIO_SIGMA = 1.8374
INVERSE_MEAN = 0.5

def test_Unc_input_output():
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

    assert b.__repr__() == "1.0 - 0.0 + 0.0"

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
        c = a/b
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
    a.set_limits([3., 4.])
    assert a.limits[0] == 3.
    assert a.limits[1] == 4.


def test_Unc_MC_algebra():
    """For symmetric uncertainties, analytic expressions exist for the mean value and standard \
    deviation of a sum/difference/product/ratio of two random variables.
    Compare the Monte-Carlo values to the analytical results, allowing \
    a maximum relative deviation of STATISTICAL_UNCERTAINTY_LIMIT
    """
    a = Unc(1., 0.1, 0.1)
    b = Unc(1., 0.1, 0.1)

    add = a + b
    sub = a - b

    # For addition and subtraction ( __sub__(), __add()__() ), the expected uncertainty of the result is sqrt(sigma_1**2 + sigma_2**2)
    assert add.mean_value >= 2. - STATISTICAL_UNCERTAINTY_LIMIT*2. and add.mean_value <= 2. + STATISTICAL_UNCERTAINTY_LIMIT*2.
    assert add.sigma_low >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and add.sigma_low <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1
    assert add.sigma_up >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and add.sigma_up <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1

    assert sub.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and sub.mean_value <= 2. + STATISTICAL_UNCERTAINTY_LIMIT*2.
    assert sub.sigma_low >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and sub.sigma_low <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1
    assert sub.sigma_up >= SQRT2*0.1 - STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1 and sub.sigma_up <= SQRT2*0.1 + STATISTICAL_UNCERTAINTY_LIMIT*SQRT2*0.1

    # For multiplication ( __mul__() ) of two normal distributed quantities with mean value == 0 and sigma == 1, the expected product distribution is K_0(|x|)/pi, where K_0 is the modified Bessel function of the second kind of order 0
    # By numerical integration, one finds that the 68.27% coverage interval is about [-0.68327, 0.68327]
    a = Unc(0., 1., 1.)
    b = Unc(0., 1., 1.)

    mult = a*b

    assert mult.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and mult.mean_value <= 0. + STATISTICAL_UNCERTAINTY_LIMIT*2.
    assert mult.sigma_low >= MULTIPLICATION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA and mult.sigma_low <= MULTIPLICATION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA
    assert mult.sigma_up >= MULTIPLICATION_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA and mult.sigma_up <= MULTIPLICATION_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*MULTIPLICATION_SIGMA

    # For the ratio ( __truediv__() ) of two normal distributed quantities with mean value == 0 and sigma == 1, the expected ratio distribution is a Cauchy distribution.
    # By numerical integration, one finds that the 68.27% coverage interval is about [-1.8374, 1.8374]
    a = Unc(0., 1., 1.)
    b = Unc(0., 1., 1.)

    ratio = a/b

    assert ratio.mean_value >= 0. - STATISTICAL_UNCERTAINTY_LIMIT*2. and ratio.mean_value <= 0. + STATISTICAL_UNCERTAINTY_LIMIT*2.
    assert ratio.sigma_low >= RATIO_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*RATIO_SIGMA and ratio.sigma_low <= RATIO_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*RATIO_SIGMA
    assert ratio.sigma_up >= RATIO_SIGMA - STATISTICAL_UNCERTAINTY_LIMIT*RATIO_SIGMA and ratio.sigma_up <= RATIO_SIGMA + STATISTICAL_UNCERTAINTY_LIMIT*RATIO_SIGMA

    # Test __rtruediv__()  which determines the division of a normal float and Unc, by calculating the inverse of a random variable
    # For this, take the expected ratio distribution for the division of two normal distributed random variables with different mean value and sigma (D.V. Hinkley, Biometrika 56, 3 (1969) 635).
    # By numerical integration (choose sigma1 << sigma2 to simulate a discrete number divided by one with uncertainty), one finds that the most probable value of the resulting distribution is 0.5
    a = Unc(1., 1., 1.)
    ratio = 1./a

    assert ratio.mean_value >= INVERSE_MEAN - STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN and ratio.mean_value <= INVERSE_MEAN + STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN 

    ratio = Unc(1., 0., 0.)/a

    assert ratio.mean_value >= INVERSE_MEAN - STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN and ratio.mean_value <= INVERSE_MEAN + STATISTICAL_UNCERTAINTY_LIMIT*INVERSE_MEAN 

    # Test __pow__() by calculating x**1, which should not change the value in the ideal case
    a = Unc(1., 1., 1.)

    power = a**1.

    assert power.mean_value >= 1. - STATISTICAL_UNCERTAINTY_LIMIT*2. and power.mean_value <= 1. + STATISTICAL_UNCERTAINTY_LIMIT*2.
    assert power.sigma_low >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.sigma_low <= 1. + STATISTICAL_UNCERTAINTY_LIMIT
    assert power.sigma_up >= 1. - STATISTICAL_UNCERTAINTY_LIMIT and power.sigma_up <= 1. + STATISTICAL_UNCERTAINTY_LIMIT

    # Test the behavior when the most probable value is not inside the shortest coverage interval. By default, the code forces the most probable value to be inside the shortest coverage interval.
    # When calculating the result in Unc.eval(), a new Unc object is created with sigma_low = most_probable - sc[0] and sigma_up = sc[1] - most_probable (sc[0] and sc[1] are the lower and upper limit of the shortest coverage interval, respectively).
    # If most_probable > sc[1] or most_probable < sc[0], a ValueError should be thrown

    # Create a distribution with the most probable value outside the shortest coverage interval 
    dist = concatenate((ones(24)*1.,
                        ones(24)*2.,
                        ones(24)*3.,
                        ones(28)*10.))

    with pytest.raises(ValueError):
        a.eval(dist, force_inside_shortest_coverage=False)

def test_Unc_float_algebra():
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

def test_Unc_exact_algebra():
    a = Unc(1., 0., 0.)
    b = Unc(1., 0., 0.)

    add = a + b
    sub = a - b
    mult = a*b
    ratio = a/b
    power = a**1.

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

    add = a + b
    sub = a - b
    mult = a*b
    ratio = a/b

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

def test_Unc_correlated_algebra():
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

    mult = a*a
    power = a**2

    assert mult.mean_value == power.mean_value
    assert mult.sigma_low == power.sigma_low
    assert mult.sigma_up == power.sigma_up

def test_Unc_functions():
    # Test exp by showing that exp(Unc(1., sigma, sigma)) gets closer to 
    # e if sigma is reduced

    a = Unc(1., 1., 1.)
    c = exp(a)
    assert c.mean_value < 1.

    a = Unc(1., 0.5, 0.5)
    c = exp(a)
    assert c.mean_value < 2.1

    a = Unc(1., 0.05, 0.05)
    c = exp(a)
    assert c.mean_value >= 2.7 and c.mean_value <= 2.8

    a = Unc(1., 0., 0.)
    c = exp(a)
    assert c.mean_value == nexp(1.)

def test_round():
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
    assert a.__repr__() == "0.05 - 0.12 + 0.32"
