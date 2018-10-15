"""Algebra for quantities with asymmetric uncertainties \
        using a Monte-Carlo uncertainty propagation method"""

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

from mc_statistics import check_num_array_argument, randn_asym

from .algebra import add, mul, power, sub, truediv
from .evaluation import evaluate
from .io import check_limit_update, check_numeric, round_digits, set_limits, set_lower_limit
from .io import set_mean_value, set_sigma_low, set_sigma_up, set_upper_limit, update_limits

class Unc:
    """Class for a quantity with asymmetric uncertainty"""

    # Count the number of instances of Unc
    n_instances = 0

    def __init__(self, mean_value, sigma_low, sigma_up, limits=None):
        try:
            self.mean_value = mean_value

            if sigma_low < 0.:
                raise ValueError("sigma_low must be >= 0.")
            if sigma_up < 0.:
                raise ValueError("sigma_up must be >= 0.")
            self.sigma_low = sigma_low
            self.sigma_up = sigma_up
            self.is_exact = bool(sigma_low == 0. and sigma_up == 0.)

            if limits is None:
                self.limits = [-inf, inf]
            else:
                self.limits = limits
            check_num_array_argument(self.limits, 2, argument_name="Limits", is_increasing=True)

        except ValueError:
            print("ValueError")
            raise

        # Initialize rounded values
        self.rounded = [self.mean_value, self.sigma_low, self.sigma_up]
        self.round_digits()

        # Set unique random number seed as the number of instances of Unc
        self.seed = Unc.n_instances
        Unc.n_instances += 1

    ###################################################
    # Input / Output
    ###################################################

    def set_mean_value(self, mean_value):
        """Set the value of mean_value

        Parameters
        ----------
        mean_value : float
            New value for mean_value
        """

        set_mean_value(self, mean_value)

    def set_sigma_low(self, sigma_low):
        """Set the value of sigma_low and check whether the new value is valid, \
                i.e. self.set_sigma_low(x) is safer than self.sigma_low = x.

        Parameters
        ----------
        sigma_low : float
            New value for sigma_low
        """

        set_sigma_low(self, sigma_low=sigma_low)

    def set_sigma_up(self, sigma_up):
        """Set the value of sigma_up and check whether the new value is valid, \
                i.e. self.set_sigma_up(x) is safer than self.sigma_up = x.

        Parameters
        ----------
        sigma_up : float
            New value for sigma_up
        """

        set_sigma_up(self, sigma_up=sigma_up)

    def set_lower_limit(self, lower_limit):
        """ Set the value of the lower limit and check whether the new value\
        is valid, i.e. self.set_lower_limit(x) is safer than
        self.lower_limit = x

        Parameters
        ----------
        lower_limit : int, float
            New value for the lower limit
        """

        set_lower_limit(self, lower_limit=lower_limit)

    def set_upper_limit(self, upper_limit):
        """ Set the value of the upper limit and check whether the new value\
        is valid, i.e. self.set_upper_limit(x) is safer than
        self.upper_limit = x

        Parameters
        ----------
        upper_limit : int, float
            New value for the upper limit
        """

        set_upper_limit(self, upper_limit=upper_limit)

    def set_limits(self, limits):
        """ Set the value of the lower and upper limits and check whether the new values\
        are valid, i.e. self.set_limits(x, y) is safer than
        self.lower_limit = x and self.upper_limit = y

        Parameters
        ----------
        limits : [float, float]
            New values for the limits
        """

        set_limits(self, limits=limits)

    def check_limit_update(self, new_limits):
        """ Check whether the new limits make sense, considering the old ones.
        For example, assume that the previous limits were [-1, 1] for a number
        Unc(0., 1., 1.) and the new ones are [10, 11].
        In order to sample points within the new limits, highly improbable values
        of the probability distribution would have to be sampled, leading to numerical
        instabilities.
        Going through the example:
        The value of the CDF of the normal distribution with mean = 0 and sigma == 1
        at x == 6 is already 0.9999999990134123, at x == 10, scipy.stats.norm.cdf simply
        displays 1.0.
        Consequently, using scipy.stats.norm.ppf(1.0) would yield infinity.

        Parameters
        ----------
        new_limits : [float, float]
            New values for the limits
        """

        check_limit_update(self, new_limits=new_limits)

    def round_digits(self):
        """ Round mean value and uncertainty limits to a sensible number of digits
        for displaying them
        The decision of how many digits to keep is made after recommendations
        by the Particle Data Group (PDG)
        """

        round_digits(self)

    def update_limits(self):
        """ If the limits for the distribution of Unc are changed, adapt mean_value,
        sigma_low and sigma_up accordingly using the same Monte-Carlo method that is
        used for all calculations
        """

        update_limits(self)

    def __repr__(self):
        return str(self.rounded[0]) + " - " + str(self.rounded[1]) + " + " + str(self.rounded[2])

    def __str__(self):
        return (str(self.rounded[0]) + "_{" + str(self.rounded[1]) + "}^{" +
                str(self.rounded[2]) + "}")

    ###################################################
    # Evaluation of results
    ###################################################

    @classmethod
    def eval(cls, rand_result, force_inside_shortest_coverage=True):
        """Evaluate the most probable value (or the mean value) and the shortest coverage interval \
of randomly sampled values.

        Parameters
        ----------
        rand_result : ndarray
            Array of random numbers
        force_inside_shortest_coverage : bool, optional
            Force the most probable value to be inside the shortest coverage interval

        Return
        ------
        result : Unc
        """

        eval_result = evaluate(rand_result=rand_result,
                               force_inside_shortest_coverage=force_inside_shortest_coverage)

        return Unc(eval_result[0], eval_result[1], eval_result[2])

    @classmethod
    def is_unc(cls, other):
        """Determine whether a given object is an instance of the class Unc

        Parameters
        ----------
        other : anything

        Return
        ------
        result : bool
            True, if other is of type Unc
            False, else
        """

        return isinstance(other, Unc)

    ###################################################
    # Algebra
    ###################################################

    def __neg__(self):
        """Switch the sign of Unc using the unary '-' operator

        Parameters
        ----------
        None

        Returns
        -------
        Unc(-self.mean_value, self.sigma_low, self.sigma_up)

        """

        return Unc(-self.mean_value, self.sigma_low, self.sigma_up)

    def __truediv__(self, other):
        """Calculate self/other

        Parameters
        ----------
        self : Unc
        other : Unc

        Returns
        -------
        self/other : Unc
        """

        truediv_result = truediv(self, other)

        return Unc(truediv_result[0], truediv_result[1], truediv_result[2])

    def __rtruediv__(self, other):
        """Calculate other/self

        Parameters
        ----------
        self : Unc
        other : int or float

        Returns
        -------
        other/self : Unc
        """

        rtruediv_result = truediv(Unc(other, 0., 0.), self)

        return Unc(rtruediv_result[0], rtruediv_result[1], rtruediv_result[2])

    def __add__(self, other):
        """Calculate self + other

        Parameters
        ----------
        self : Unc
        other : Unc

        Returns
        -------
        self + other : Unc
        """

        add_result = add(self, other)

        return Unc(add_result[0], add_result[1], add_result[2])

    def __radd__(self, other):
        """Calculate other + self

        Parameters
        ----------
        self : Unc
        other : float or int

        Returns
        -------
        other + self : Unc
        """

        radd_result = add(self, other)

        return Unc(radd_result[0], radd_result[1], radd_result[2])

    def __sub__(self, other):
        """Calculate self - other

        Parameters
        ----------
        self : Unc
        other : Unc

        Returns
        -------
        self - other : Unc
        """

        sub_result = sub(self, other)

        return Unc(sub_result[0], sub_result[1], sub_result[2])

    def __rsub__(self, other):
        """Calculate other - self

        Parameters
        ----------
        self : Unc
        other : float or int

        Returns
        -------
        other - self : Unc
        """

        rsub_result = sub(self, other)

        return Unc(rsub_result[0], rsub_result[1], rsub_result[2])

    def __mul__(self, other):
        """Calculate self*other

        Parameters
        ----------
        self : Unc
        other : Unc

        Returns
        -------
        self*other : Unc
        """

        mul_result = mul(self, other)

        return Unc(mul_result[0], mul_result[1], mul_result[2])

    def __rmul__(self, other):
        """Calculate other*self

        Parameters
        ----------
        self : Unc
        other : float or int

        Returns
        -------
        self*other : Unc
        """

        rmul_result = mul(self, other)

        return Unc(rmul_result[0], rmul_result[1], rmul_result[2])

    def __pow__(self, other):
        """Calculate self**other

        Parameters
        ----------
        self : Unc
        other: Unc

        Returns
        -------
        self**other: Unc
        """

        pow_result = power(self, other)

        return Unc(pow_result[0], pow_result[1], pow_result[2])
