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

from mc_statistics import randn_asym

from .evaluation import evaluate
from .io import check_numeric

def truediv(self, other):
    """Implementation of Unc.__truediv__()"""

    check_numeric(self, other)

    if isinstance(other, (int, float)):
        return [self.mean_value/other, self.sigma_low/other, self.sigma_up/other]

    if self.seed == other.seed:
        return [1., 0., 0.]

    if other.is_exact:
        return [self.mean_value/other.mean_value, self.sigma_low/other.mean_value,
                self.sigma_up/other.mean_value]

    rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                            limits=self.limits, random_seed=other.seed)

    if self.is_exact:
        rand_result = self.mean_value/rand_other

    else:
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed)
        rand_result = rand_self/rand_other

    return evaluate(rand_result, force_inside_shortest_coverage=True)

def add(self, other):
        """Implementation of Unc.__add__()"""

        check_numeric(self, other)

        if isinstance(other, (int, float)):
            return [self.mean_value + other, self.sigma_low, self.sigma_up]

        if self.seed == other.seed:
            return [2.*self.mean_value, 2.*self.sigma_low, 2.*self.sigma_up]

        if other.is_exact:
            return [self.mean_value + other.mean_value, self.sigma_low, self.sigma_up]

        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=self.limits, random_seed=other.seed)

        rand_result = rand_self + rand_other

        return evaluate(rand_result)


def sub(self, other):
        """Implementation of Unc.__sub__()"""

        check_numeric(self, other)

        if isinstance(other, (int, float)):
            return [self.mean_value - other, self.sigma_low, self.sigma_up]

        if self.seed == other.seed:
            return [0., 0., 0.]

        if other.is_exact:
            return [self.mean_value - other.mean_value, self.sigma_low, self.sigma_up]

        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=self.limits, random_seed=other.seed)

        rand_result = rand_self - rand_other

        return evaluate(rand_result)

def mul(self, other):
        """Implementation of Unc.__mul__()"""

        check_numeric(self, other)

        if isinstance(other, (int, float)):
            return [self.mean_value*other, self.sigma_low*other, self.sigma_up*other]

        if other.is_exact:
            return [self.mean_value*other.mean_value, self.sigma_low*other.mean_value,
                        self.sigma_up*other.mean_value]

        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, [other.sigma_low, other.sigma_up],
                                limits=self.limits, random_seed=other.seed)

        rand_result = rand_self*rand_other

        return evaluate(rand_result)

def power(self, other):
        """Implementation of Unc.__pow__()"""

        check_numeric(self, other)
    
        if self.is_exact:
            return [self.mean_value**other.mean_value, 0., 0.]
    
        rand_self = randn_asym(self.mean_value, [self.sigma_low, self.sigma_up],
                               limits=self.limits, random_seed=self.seed)
    
        rand_result = rand_self**exponent
    
        return evaluate(rand_result)
