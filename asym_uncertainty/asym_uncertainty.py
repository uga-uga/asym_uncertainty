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

from numpy import array

from mc_statistics import check_num_array_argument

from .algebra import add, mul, power, rpower, sub, truediv
from .evaluation import evaluate
from .io import check_limit_update, check_numeric, round_digits, sample_random_numbers
from .io import set_limits, set_lower_limit, set_mean_value, set_sigma_low, set_sigma_up
from .io import set_upper_limit, update_limits

class Unc:
    """Class for a quantity with asymmetric uncertainty

    An object of Unc represents a number x with asymmetric uncertainty

    x = x_mean - dx_low + dx_up

    for example

    x = 1.0 + 0.5 - 0.3

    The interval [dx_low, dx_up] is assumed to correspond to the 1-sigma interval of a
    symmetric normal distribution N(x_mean, sigm), i.e. it contains about 68.27 % of the
    probability distribution of x.
    Unc implements mathematical operators to perform mathematical operations on numbers with
    asymmetric uncertainties.
    In a mathematical operation, the probability distribution for x is approximated as a
    discontinuous combination of two normal distributions

             | N(x_mean, dx_low), x <  x_mean
    PDF(x) = |
             | N(x_mean, dx_up ), x >= x_mean

    The result of an operation z = O(x,y) on two numbers x and y with asymmetric uncertainty is
    determined by sampling N random values x_rand and y_rand from their probability distributions
    and calculating the result z_rand N times.
    The given z_mean, dz_low and dz_up will be the most probable value of the resulting
    distribution of z_rand and the limits of the shortest coverage interval (the shortest interval
    that contains 68.27 % percent of the sampled values z_rand)

    The probability distribution of a number with uncertainties can also be confined to a certain
    interval, for example if only positive numbers are allowed in a calculation.

    Note
    ----
    Assume the following attributes to be private members of Unc, i.e. do not change their values
    like

    Unc.limits[0] = x

    The example above is especially critical, because changing the limits has an impact on several
    other attributes of the Unc object (mean_value, sigma_low, sigma_up, is_exact, rounded).
    Use the correponding set* methods, which take care of all the interdependencies between the
    attributes.

    Attributes
    ----------
    mean_value: float
        Most probable value of x.
    sigma_low: float
        mean_value - sigma_low is the lower limit of the shortest coverage interval
    sigma_up: float
        mean_value + sigma_up is the upper limit of the shortest coverage interval

    is_exact: bool
        True, if sigma_low = sigma_up = 0. Used to simplify calculations, because if exact numbers
        appear, no random sampling is needed.
    limits: [float, float]
        Limits of the probability distribution. Randomly sampled values x_rand will only be inside
        inside [limits[0], limits[1]].

    rounded: [float, float, float]
        Rounded values of mean_value, sigma_low and sigma_up according to the rounding rules
        of the Particle Data Group

    seed: int
        Static variable that counts the number of Unc objects created so far and seeds the random
        number generator of x. Giving each number x a fixed seed makes it possible to introduce
        correlations in calculations, for example in a calculation like z = x/(1+x) where
        x appears several times.

    store: bool
        If True, a numpy array of the values x_rand is stored in the Unc object. Furthermore, the
        resulting Unc object from a calculation will also store the randomly sampled values
        from which its mean and shortest coverage interval were determined.
    random_values: numpy array
        Array of randomly sampled numbers from the probability distribution of Unc. The number of
        values is given by the settings of the imported mc_statistics package.
    """

    # Count the number of instances of Unc
    n_instances = 0

    def __init__(self, mean_value, sigma_low, sigma_up, limits=None, store=False, random_values=array([0.])):
        """Initialization of members of Unc

        See the class docstring of Unc for the meaning of the member variables
        that can be set with __init__().

        Parameters
        ----------
        mean_value: float
        sigma_low: float
        sigma_up: float
        limits: [float, float]
        store: bool
        random_values: numpy array
        """
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

        self.store = store
        self.random_values = random_values

        if self.store and len(random_values) < 2:
            self.sample_random_numbers()

    ###################################################
    # Input / Output
    ###################################################

    def sample_random_numbers(self):
        """Sample random numbers from the distribution of Unc

        Using the mean value, the shortest coverage interval and the limits, sample random
        numbers and store them in the random_values member variable.
        """

        sample_random_numbers(self)

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

    def eval(self, rand_result, force_inside_shortest_coverage=True):
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

        if self.store:
            return Unc(eval_result[0][0], eval_result[0][1], eval_result[0][2],
                       random_values=eval_result[1])
        return Unc(eval_result[0][0], eval_result[0][1], eval_result[0][2])

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

        if self.store:
            return Unc(truediv_result[0][0], truediv_result[0][1], truediv_result[0][2], 
                       random_values=truediv_result[1])

        return Unc(truediv_result[0][0], truediv_result[0][1], truediv_result[0][2])

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

        check_numeric(self, other)

        rtruediv_result = truediv(Unc(other, 0., 0.), self)

#        if self.store:
#            return Unc(rtruediv_result[0][0], rtruediv_result[0][1], rtruediv_result[0][2],
#                       random_values=rtruediv_result[0][1])
        return Unc(rtruediv_result[0][0], rtruediv_result[0][1], rtruediv_result[0][2])

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

        if self.store:
            return Unc(add_result[0][0], add_result[0][1], add_result[0][2],
                       random_values=add_result[1])
        return Unc(add_result[0][0], add_result[0][1], add_result[0][2])

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

#        if self.store:
#            return Unc(radd_result[0][0], radd_result[0][1], radd_result[0][2], 
#                       random_values=radd_result[1])
        return Unc(radd_result[0][0], radd_result[0][1], radd_result[0][2])

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

        if self.store:
            return Unc(sub_result[0][0], sub_result[0][1], sub_result[0][2], 
                       random_values=sub_result[1])
        return Unc(sub_result[0][0], sub_result[0][1], sub_result[0][2])

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

#        if self.store:
#            return Unc(rsub_result[0][0], rsub_result[0][1], rsub_result[0][2],
#                       random_values=rsub_result[1])
        return Unc(rsub_result[0][0], rsub_result[0][1], rsub_result[0][2])

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

        if self.store:
            return Unc(mul_result[0][0], mul_result[0][1], mul_result[0][2],
                       random_values=mul_result[1])
        return Unc(mul_result[0][0], mul_result[0][1], mul_result[0][2])

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

#        if self.store:
#            return Unc(rmul_result[0][0], rmul_result[0][1], rmul_result[0][2],
#                       random_values=rmul_result[1])
        return Unc(rmul_result[0][0], rmul_result[0][1], rmul_result[0][2])

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

#        if self.store:
#            return Unc(pow_result[0][0], pow_result[0][1], pow_result[0][2],
#                       random_values=pow_result[0])
        return Unc(pow_result[0][0], pow_result[0][1], pow_result[0][2])

    def __rpow__(self, other):
        """Calculate other**self

        Parameters
        ----------
        self : Unc
        other : float or int

        Returns
        -------
        self*other : Unc
        """

        rpow_result = rpower(self, other)

#        if self.store:
#            return Unc(rpow_result[0][0], rpow_result[0][1], rpow_result[0][2],
#                       random_values=rpow_result[1])
        return Unc(rpow_result[0][0], rpow_result[0][1], rpow_result[0][2])
