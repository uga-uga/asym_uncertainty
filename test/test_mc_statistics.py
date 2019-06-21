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
from numpy import array, concatenate, inf, linspace, maximum, mean, std
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import beta 

from asym_uncertainty import cdf, check_num_array_argument, chi2, randn_asym, shortest_coverage

def test_cdf():
    # Test CDF with a simple array
    test_array = array([3., 1., 2.]) 
    cdf_result = cdf(test_array)

    assert cdf_result[0][0] == 1.
    assert cdf_result[0][1] == 2.
    assert cdf_result[0][2] == 3.
    assert cdf_result[1][0] == 0.
    assert cdf_result[1][1] == 0.5
    assert cdf_result[1][2] == 1.

def test_shortest_coverage():
    # Test shortest coverage with sampled values from the CDF of a normal distribution. 
    # In this case, the shortest coverage interval should be identical, within the uncertainty due to the finite number of samples, to the interval [-sigma + mu, sigma + mu]

    x = linspace(0., 1., int(1e7))
    cdf_input = [norm.ppf(x, 0., 1.), x]

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)
    
    assert sc_result[0] - sc_result[2] <= -1. and sc_result[0] + sc_result[2] >= -1. 
    assert sc_result[1] - sc_result[3] <=  1. and sc_result[1] + sc_result[3] >=  1. 
    # Test without uncertainty estimate
    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=False)

    assert sc_result[0] >= -1.1 and sc_result[0] <= -0.9

    # Test with a distribution where the shortest coverage interval starts at the lower edge of the range -> Exponential distribution

    cdf_input = [expon.ppf(x, 0., 1.), x]

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)
    
    assert sc_result[0] - sc_result[2] <= 0. and sc_result[0] + sc_result[2] >= 0.

    # Turn the range of values of the exponential distribution from [0, infinity] to [-infinity, 0] to test the behavior for negative numbers

    cdf_input = [-expon.ppf(x, 0., 1.), x]

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)
    
    assert sc_result[1] - sc_result[3] <= 0. and sc_result[1] + sc_result[3] >= 0.

    # Use a beta distribution to test the case when the upper limit of the shortest coverage is at the upper limit of the range of values

    cdf_input = [beta.ppf(x, 5, 1, 0., 1.), x]

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)
    
    assert sc_result[1] - sc_result[3] <= 1. and sc_result[1] + sc_result[3] >= 1.

    # Test with a uniform distribution, because it has another uncertainty estimate

    cdf_input = [x, x] 

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)

    assert sc_result[0] - sc_result[2] <= 0. and sc_result[0] + sc_result[2] >= 0.

    # The same for negative values

    cdf_input = [-x, x] 

    sc_result = shortest_coverage(cdf_input, coverage_percent=68.27, uncertainty_estimate=True)

    assert sc_result[1] - sc_result[3] <= 0. and sc_result[1] + sc_result[3] >= 0.

def test_chi2():
    assert chi2(array([0., 1.]), array([1., 1.]), array([1., 0.]), degrees_of_freedom=1) == 2.

def test_randn_asym():
    # Generate number with a symmetric distribution and check whether mean and standard deviation agree with a normal distribution
    random_values = randn_asym(0., [1., 1.], conserve_mean_value=False)
    mean_value = mean(random_values)
    standard_deviation = std(random_values)    
    assert -0.1 <= mean_value and mean_value <= 0.1
    assert 0.9 <= standard_deviation and standard_deviation <= 1.1

    # Check whether conserve_mean_value works when sigma1 != sigma2
    mean_value = mean(randn_asym(0., [1., 2.], conserve_mean_value=True))
    assert -0.1 <= mean_value and mean_value <= 0.1

    # Check the warning when conserve_mean_value is activated and limits are set
    with pytest.raises(RuntimeWarning):
        mean_value = mean(randn_asym(0., [1., 2.], limits=[-1., 1.], conserve_mean_value=True))

    # Check distribution with limits
    random_values = randn_asym(0., [1., 1.], limits=[0., inf], conserve_mean_value=False)
    assert random_values.all() >= 0.
    random_values_symmetric = concatenate((random_values, -random_values))
    # In order to check the mean value, symmetrize the values by concatenating them with their own negatives
    mean_value = mean(random_values_symmetric)
    standard_deviation = std(random_values_symmetric)
    assert -0.1 <= mean_value and mean_value <= 0.1
    assert 0.9 <= standard_deviation and standard_deviation <= 1.1

    # Since the assumed probability distribution is not continuous, the random number
    # generator needs to distinguish different cases where the mean value is outside
    # or inside the given limits.
    # Check with mean value == -1 < limits[0] The mean value of the truncated distribution below should be ~0.525
    random_values = randn_asym(-1., [1., 1.], limits=[0., inf], conserve_mean_value=False)
    mean_value = mean(random_values)

    assert 0.51 <= mean_value and mean_value <= 0.53

    # Check with mean value == 1 > limits[1]. The mean value of the truncated distribution below should be ~-0.525
    random_values = randn_asym(1., [1., 1.], limits=[-inf, 0.], conserve_mean_value=False)
    mean_value = mean(random_values)

    assert -0.51 >= mean_value and mean_value >= -0.53

    # Check with mean value == 1 > limits[0]. The mean value of the truncated distribution below should be ~-1.287
    random_values = randn_asym(1., [1., 1.], limits=[0., inf], conserve_mean_value=False)
    mean_value = mean(random_values)

    assert 1.28 <= mean_value and mean_value <= 1.295

    # Check the input of random number seeds
    with pytest.raises(ValueError):
        randn_asym(1., [1., 1.], conserve_mean_value=False, random_seed=-1)
    with pytest.raises(ValueError):
        randn_asym(1., [1., 1.], conserve_mean_value=False, random_seed='a')


    # Check the setting of the random number seed
    random_values1 = randn_asym(1., [1., 1.], conserve_mean_value=False, random_seed=1)
    random_values2 = randn_asym(1., [1., 1.], conserve_mean_value=False, random_seed=1)
    random_values3 = randn_asym(1., [1., 1.], conserve_mean_value=False, random_seed=2)
    assert (random_values1 == random_values2).all()
    assert not (random_values1.all() == random_values3).all()

def test_randn_asym_input():
    # randn_asym takes two arrays as arguments, sigma and limits. Both need to have a special format.
    # In order to ensure the correct format, the function check_num_array_argument exists

    with pytest.raises(ValueError):
        check_num_array_argument([-1], 2)
    with pytest.raises(ValueError):
        check_num_array_argument(['a'], 1)
    with pytest.raises(ValueError):
        check_num_array_argument([-1], 1, is_positive=True)
    with pytest.raises(ValueError):
        check_num_array_argument([1, -1], 2, is_increasing=True)

    # Check whether randn_asym really returns a scalar if n_random == 1
    # scalar: len(1) -> TypeError
    # vector: len([1]) == 1
    with pytest.raises(TypeError):
        len(randn_asym(1, [0.5, 0.5], n_random=1))
    with pytest.raises(TypeError):
        len(randn_asym(1, [0.5, 0.5], limits=[2., inf], n_random=1))
    with pytest.raises(TypeError):
        len(randn_asym(1, [0.5, 0.5], limits=[-inf, 0.], n_random=1))
    with pytest.raises(TypeError):
        len(randn_asym(1, [0.5, 0.5], limits=[0., 2.], n_random=1))
