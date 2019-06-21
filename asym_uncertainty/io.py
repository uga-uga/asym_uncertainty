"""Input/Output for the Unc class"""

import warnings

from numpy import array, absolute, sort, extract, floor, log10
from numpy import minimum as nminimum
from numpy import round as nround
from numpy.random import uniform

from scipy.interpolate import interp1d

from .mc_statistics import cdf, check_num_array_argument, randn_asym

from .evaluation import evaluate

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

def check_limit_update(self, new_limits):
    """Implementation of Unc.check_limit_update()"""

    try:
        if new_limits[0] >= self.limits[1] or new_limits[1] <= self.limits[0]:
            raise ValueError("New limits are outside of old limits.")
    except ValueError:
        raise

    if new_limits[0] < self.limits[0] or new_limits[1] > self.limits[1]:
        warnings.warn("New limits are larger than old limits, \
re-sampling stored random values may give unexpected results.", RuntimeWarning)

def check_numeric(self, other):
    """Implementation of Unc.check_numeric()"""
    try:
        if isinstance(other, (int, float)):
            return True
        if self.is_unc(other):
            return True
        raise ValueError("Right-hand operand must be either a built-in\
                             numerical type or Unc")

    except ValueError:
        print("ValueError")
        raise

def round_digits(self):
    """Implementation of Unc.round_digits()"""
    arr = array([self.mean_value, self.sigma_low, self.sigma_up])

    # Safety measure: if the absolute mean value is much smaller than the uncertainty,
    # simply set it to zero
    if arr.all() > 0. and absolute(arr[0])/nminimum(arr[1], arr[2]) < 10**-1:
        arr[0] = 0.

    # If both the mean value and both uncertainties are zero, treat it as  a special case.
    # In all other cases, the smallest nonzero value of [mean_value, sigma_low, sigma_up]
    # determines the digits of the rounded values.
    arr_sort = sort(arr)
    nonzero = extract(arr_sort > 0., arr_sort)
    if len(nonzero) is not 0:
        if self.sigma_low == self.sigma_up == 0.:
            arr_round = [self.mean_value, 0., 0.]

        else:
            first_digit = floor(log10(absolute(nonzero[0])))
            # Make a decision on the number of displayed digits based on a recommendation
            # by the PDG
            first_three_digits = nround(nonzero[0]*10**(-first_digit+2))

            if 100 <= first_three_digits <= 354:
                rounding_digits = 1
            elif 355 <= first_three_digits <= 949:
                rounding_digits = 0
            else:
                rounding_digits = 1

            arr_round = (nround(arr*10**(-first_digit+rounding_digits))/
                         10**(-first_digit+rounding_digits))

            self.rounded = arr_round

    # Nothing else needed. If mean_value == sigma_low == sigma_up, the default value of
    # self.rounded will be [0., 0., 0.] anyway.

def sample_random_numbers(self):
    """Implementation of Unc.sample_random_numbers()"""

    if len(self.random_values) > 1:
        cdf_generating_random_values = (extract((self.random_values >= self.limits[0])*
                                                (self.random_values <= self.limits[1]),
                                                self.random_values))

        random_value_fraction = (float(len(cdf_generating_random_values))/
                                 float(len(self.random_values)))

        try:
            if len(cdf_generating_random_values) < 2:
                raise ValueError("Trying to re-sample random numbers. \
    Within the current limits, no CDF can be reconstructed from self.random_values, since \
    less than two of them are inside the new limits.")

        except ValueError:
            raise

        if len(cdf_generating_random_values) < 1000:
            warnings.warn("Trying to re-sample random numbers. \
Within the current limits, less than 1000 values will be available to construct a CDF. \
This may lead to unintended behavior. Check whether the limits make sense or try increasing \
n_random.", RuntimeWarning)
        if random_value_fraction < 0.01:
            warnings.warn("Trying to re-sample random numbers. \
Within the current limits, less than 1 percent of the previous values will be available to \
construct a CDF. This may lead to unintended behavior. Check whether the limits make sense or \
try increasing n_random.", RuntimeWarning)

        cdf_discrete = cdf(cdf_generating_random_values)
        inverse_cdf = interp1d(cdf_discrete[1], cdf_discrete[0])
        self.random_values = inverse_cdf(uniform(0., 1., size=self.n_random))

    else:
        self.random_values = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                                        limits=self.limits, random_seed=self.seed,
                                        n_random=self.n_random)
    eval_result = evaluate(self.random_values, force_inside_shortest_coverage=True)
    self.set_mean_value(eval_result[0][0])
    self.set_sigma_low(eval_result[0][1])
    self.set_sigma_up(eval_result[0][2])

def set_limits(self, limits):
    """Implementation of Unc.set_limits()"""

    check_num_array_argument(limits, 2, argument_name="Limits",
                             is_increasing=True)
    self.check_limit_update(limits)

    self.limits = limits
    if not self.is_exact:
        self.sample_random_numbers()

def set_lower_limit(self, lower_limit):
    """Implementation of Unc.set_lower_limit()"""

    check_num_array_argument([lower_limit, self.limits[1]], 2, argument_name="Limits",
                             is_increasing=True)
    self.check_limit_update([lower_limit, self.limits[1]])

    self.limits[0] = lower_limit
    if not self.is_exact:
        self.sample_random_numbers()

def set_mean_value(self, mean_value):
    """Implementation of Unc.set_mean_value()"""
    self.mean_value = mean_value
    self.round_digits()

def set_n_random(self, n_random):
    """Implementation of Unc.set_n_random()"""
    try:
        if not isinstance(n_random, int):
            raise ValueError("n_random must be an integer.")
        if n_random < 2:
            raise ValueError("n_random must be > 1 and should be much larger than that.")

        self.n_random = n_random
        # If storage of sampled values is desired, update the number
        # of stored random numbers, either by truncating the existing
        # set, or by sampling anew.
        if self.store:
            if len(self.random_values) >= n_random:
                self.random_values = self.random_values[0:n_random]
            else:
                # This condition is there to ignore calls of set_n_random by the constructor
                # of Unc. If the constructor is called with store=True and the default value
                # of random_values, which is a length-1 numpy array, then the resizing of the
                # array is intentional, and no warning is needed.
                if len(self.random_values) > 1:
                    warnings.warn("Requested n_random (%i) is larger than stored number of \
random values (%i). Sampling new set of random values assuming \
an asymmetric normal distribution." %
                                  (n_random, len(self.random_values)), UserWarning)
                self.sample_random_numbers()

    except ValueError:
        print("ValueError")
        raise

def set_sigma_low(self, sigma_low):
    """Implementation of Unc.set_sigma_low()"""
    try:
        if sigma_low < 0.:
            raise ValueError("sigma_low must be >= 0.")

        self.sigma_low = sigma_low
        self.is_exact = bool(self.sigma_low == 0. and self.sigma_up == 0.)
        self.round_digits()
    except ValueError:
        print("ValueError")
        raise

def set_sigma_up(self, sigma_up):
    """Set the value of sigma_up and check whether the new value is valid, \
            i.e. self.set_sigma_up(x) is safer than self.sigma_up = x.

    Parameters
    ----------
    sigma_up : float
        New value for sigma_up
    """
    try:
        if sigma_up < 0.:
            raise ValueError("sigma_up must be >= 0.")

        self.sigma_up = sigma_up
        self.is_exact = bool(self.sigma_low == 0. and self.sigma_up == 0.)
        self.round_digits()
    except ValueError:
        print("ValueError")
        raise

def set_upper_limit(self, upper_limit):
    """Implementation of Unc.set_upper_limit()"""

    check_num_array_argument([self.limits[0], upper_limit], 2, argument_name="Limits",
                             is_increasing=True)
    self.check_limit_update([self.limits[0], upper_limit])

    self.limits[1] = upper_limit
    if not self.is_exact:
        self.sample_random_numbers()
