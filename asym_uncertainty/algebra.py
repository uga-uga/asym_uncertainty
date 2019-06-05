"""Definition of algebraic operators for the Unc class"""

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

import warnings

from numpy import array

from mc_statistics import randn_asym

from .evaluation import evaluate
from .io import check_numeric

def add(self, other):
    """Implementation of Unc.__add__()"""

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True

    if isinstance(other, (int, float)):
        return [([self.mean_value + other, self.sigma_low, self.sigma_up],
                 self.random_values+other),
                store_rand_result]

    if other.store:
        store_rand_result = True

    if self.seed == other.seed:
        return [([2.*self.mean_value, 2.*self.sigma_low, 2.*self.sigma_up], array([0.])),
                store_rand_result]

    if other.is_exact:
        return [([self.mean_value + other.mean_value, self.sigma_low, self.sigma_up],
                 self.random_values + other.mean_value),
                store_rand_result]
    if self.is_exact:
        return [([self.mean_value + other.mean_value, other.sigma_low, other.sigma_up],
                 self.mean_value + other.random_values), store_rand_result]

    if self.store:
        rand_self = self.random_values
    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed,
                               n_random=self.n_random)
    if self.store:
        rand_other = other.random_values
    else:
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=other.limits, random_seed=other.seed,
                                n_random=other.n_random)

    common_array_size = array_size_min(len(rand_self), len(rand_other))
    rand_result = (rand_self[0:common_array_size]+
                   rand_other[0:common_array_size])

    return [evaluate(rand_result), store_rand_result]

def array_size_min(array_1_length, array_2_length):
    """Given two array sizes, warn if they are not equal. Always return the smaller value.

    Parameters
    ----------
    array_1_length, array_2_length : int
        Two integer numbers which are assumed to be lengths of arrays

    Returns
    -------
        int
        The minimum of array_1_length and array_2_length
    """
    if array_1_length != array_2_length:
        warnings.warn("Truncated one array of random numbers due to array size mismatch.",
                      UserWarning)
    return min(array_1_length, array_2_length)

def mul(self, other):
    """Implementation of Unc.__mul__()"""

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True

    if isinstance(other, (int, float)):
        return [([self.mean_value*other, self.sigma_low*other, self.sigma_up*other],
                 self.random_values*other), store_rand_result]

    if other.store:
        store_rand_result = True

    if other.is_exact:
        return [([self.mean_value*other.mean_value, self.sigma_low*other.mean_value,
                  self.sigma_up*other.mean_value], self.random_values*other.mean_value),
                store_rand_result]
    if self.is_exact:
        return [([self.mean_value*other.mean_value, other.sigma_low*self.mean_value,
                  other.sigma_up*self.mean_value], self.mean_value*other.random_values),
                store_rand_result]


    if self.store:
        rand_self = self.random_values
    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed,
                               n_random=self.n_random)
    if other.store:
        rand_other = other.random_values
    else:
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=other.limits, random_seed=other.seed,
                                n_random=other.n_random)

    common_array_size = array_size_min(len(rand_self), len(rand_other))
    rand_result = (rand_self[0:common_array_size]*
                   rand_other[0:common_array_size])

    return [evaluate(rand_result), store_rand_result]

def power(self, other):
    """Implementation of Unc.__pow__()"""

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True

    if isinstance(other, (int, float)):
        if self.is_exact:
            return [([self.mean_value**other, 0., 0.],
                     array([self.mean_value**other])),
                    store_rand_result]

        if self.store:
            rand_self = self.random_values
        else:
            rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                                   limits=self.limits, random_seed=self.seed,
                                   n_random=self.n_random)
        rand_result = (rand_self**other)
        return [evaluate(rand_result), store_rand_result]

    if other.store:
        store_rand_result = True

    if self.is_exact:
        if other.is_exact:
            return [([self.mean_value**other.mean_value, 0., 0.],
                     array([self.mean_value**other.mean_value])),
                    store_rand_result]

        if other.store:
            rand_other = other.random_values
        else:
            rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                    limits=other.limits, random_seed=other.seed,
                                    n_random=other.n_random)

        rand_result = self.mean_value**rand_other

        return [evaluate(rand_result), store_rand_result]

    if self.store:
        rand_self = self.random_values
    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed,
                               n_random=self.n_random)

    if other.store:
        rand_other = other.random_values
    else:
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=other.limits, random_seed=other.seed,
                                n_random=other.n_random)

    common_array_size = array_size_min(len(rand_self), len(rand_other))
    rand_result = (rand_self[0:common_array_size]**
                   rand_other[0:common_array_size])

    return [evaluate(rand_result), store_rand_result]

def rpower(self, other):
    """Implementation of Unc.__pow__()"""

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True
        rand_self = self.random_values
    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed,
                               n_random=self.n_random)

    rand_result = other**rand_self

    return [evaluate(rand_result), store_rand_result]

def sub(self, other, rsub=False):
    """Implementation of Unc.__sub__()

    This function has an additional flag which indicates whether sub() was called by the
    __sub__() or the __rsub__() method of the Unc class.
    This is important in the case when an int or float number is subtracted.
    """

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True

    if isinstance(other, (int, float)):
        if rsub:
            return [([other - self.mean_value, self.sigma_low, self.sigma_up],
                     other-self.random_values),
                    store_rand_result]
        return [([self.mean_value - other, self.sigma_low, self.sigma_up],
                 self.random_values - other),
                store_rand_result]

    if other.store:
        store_rand_result = True

    if self.seed == other.seed:
        return [([0., 0., 0.], array([0.])), store_rand_result]

    if other.is_exact:
        return [([self.mean_value - other.mean_value, self.sigma_low, self.sigma_up],
                 self.random_values - other.mean_value), store_rand_result]
    if self.is_exact:
        return [([self.mean_value - other.mean_value, other.sigma_low, other.sigma_up],
                 self.mean_value - other.random_values), store_rand_result]

    if self.store:
        rand_self = self.random_values
    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed,
                               n_random=self.n_random)
    if other.store:
        rand_other = other.random_values
    else:
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=other.limits, random_seed=other.seed,
                                n_random=other.n_random)

    common_array_size = array_size_min(len(rand_self), len(rand_other))
    rand_result = (rand_self[0:common_array_size] -
                   rand_other[0:common_array_size])

    return [evaluate(rand_result), store_rand_result]

def truediv(self, other):
    """Implementation of Unc.__truediv__()"""

    store_rand_result = False
    check_numeric(self, other)

    if self.store:
        store_rand_result = True

    if isinstance(other, (int, float)):
        return [([self.mean_value/other, self.sigma_low/other, self.sigma_up/other],
                 self.random_values/other),
                store_rand_result]

    if self.seed == other.seed:
        return [([1., 0., 0.], array([1.])), store_rand_result]

    if other.is_exact:
        return [([self.mean_value/other.mean_value, self.sigma_low/other.mean_value,
                  self.sigma_up/other.mean_value], self.random_values/other.mean_value),
                store_rand_result]

    if other.store:
        rand_other = other.random_values
        store_rand_result = True
    else:
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=other.limits, random_seed=other.seed,
                                n_random=other.n_random)

    if self.is_exact:
        if self.mean_value == 0.:
            return [([0., 0., 0.], array([0.])), store_rand_result]
        rand_result = self.mean_value/rand_other

    else:
        if self.store:
            store_rand_result = True
            rand_self = self.random_values
        else:
            rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                                   limits=self.limits, random_seed=self.seed,
                                   n_random=self.n_random)
        common_array_size = array_size_min(len(rand_self), len(rand_other))
        rand_result = (rand_self[0:common_array_size]/
                       rand_other[0:common_array_size])

    return [evaluate(rand_result, force_inside_shortest_coverage=True), store_rand_result]
