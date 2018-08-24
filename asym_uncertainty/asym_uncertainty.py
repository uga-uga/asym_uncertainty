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

from numpy import argmax, extract, histogram
from mc_statistics import cdf, randn_asym, shortest_coverage

class Unc:
    """Class for a quantity with asymmetric uncertainty"""

    # Count the number of instances of Unc
    n_instances = 0

    def __init__(self, mean_value, sigma_low, sigma_up):
        try:
            self.mean_value = mean_value

            if sigma_low < 0.:
                raise ValueError("sigma_low must be >= 0.")
            if sigma_up < 0.:
                raise ValueError("sigma_up must be >= 0.")
            self.sigma_low = sigma_low
            self.sigma_up = sigma_up
            self.is_exact = bool(sigma_low == 0. and sigma_up == 0.)

        except ValueError:
            print("ValueError")
            raise

        # Set unique random number seed as the number of instances of Unc
        self.seed = Unc.n_instances
        Unc.n_instances += 1

    def set_mean_value(self, mean_value):
        """Set the value of mean_value

        Parameters
        ----------
        mean_value : float
            New value for mean_value
        """
        self.mean_value = mean_value

    def set_sigma_low(self, sigma_low):
        """Set the value of sigma_low and check whether the new value is valid, \
                i.e. self.set_sigma_low(x) is safer than self.sigma_low = x.

        Parameters
        ----------
        sigma_low : float
            New value for sigma_low
        """
        try:
            if sigma_low < 0.:
                raise ValueError("sigma_low must be >= 0.")

            self.sigma_low = sigma_low
            self.is_exact = bool(self.sigma_low == 0. and self.sigma_up == 0.)
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
        except ValueError:
            print("ValueError")
            raise

    def __repr__(self):
        return str(self.mean_value) + " - " + str(self.sigma_low) + " + " + str(self.sigma_up)

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
        result : [float, float, float]
            [most probable, <m> - lower limit of shortest coverage interval, \
upper limit of shortest coverage interval - <m>]
        """

        s_cov = shortest_coverage(cdf(rand_result))
        if force_inside_shortest_coverage:
            hist, bins = histogram(
                extract((rand_result >= s_cov[0])*(rand_result <= s_cov[1]), rand_result),
                bins="sqrt")
        else:
            hist, bins = histogram(rand_result, bins="sqrt")

        most_probable = bins[argmax(hist)]

        return Unc(most_probable, most_probable - s_cov[0], s_cov[1] - most_probable)

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

        try:
            if not isinstance(other, (int, float, Unc)):
                raise ValueError("Right-hand operand must be either a built-in\
                                 numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

        if isinstance(other, (int, float)):
            return Unc(self.mean_value/other, self.sigma_low/other, self.sigma_up/other)

        if self.seed == other.seed:
            return Unc(1., 0., 0.)

        if other.is_exact:
            return (Unc(self.mean_value/other.mean_value, self.sigma_low/other.mean_value,
                        self.sigma_up/other.mean_value))

        rand_other = randn_asym(other.mean_value, other.sigma_low, other.sigma_up,
                                random_seed=other.seed)

        if self.is_exact:
            rand_result = self.mean_value/rand_other

        else:
            rand_self = randn_asym(self.mean_value, self.sigma_low, self.sigma_up,
                                   random_seed=self.seed)
            rand_result = rand_self/rand_other

        return self.eval(rand_result, force_inside_shortest_coverage=True)

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

        try:
            if isinstance(other, (int, float)):
                return Unc(other, 0., 0.)/Unc(self.mean_value, self.sigma_low, self.sigma_up)

            raise ValueError("Left-hand operand must be either a built-in numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

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

        try:
            if not isinstance(other, (int, float, Unc)):
                raise ValueError("Right-hand operand must be either a built-in\
                                 numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

        if isinstance(other, (int, float)):
            return Unc(self.mean_value + other, self.sigma_low, self.sigma_up)

        if self.seed == other.seed:
            return 2.*Unc(self.mean_value, self.sigma_low, self.sigma_up)

        if other.is_exact:
            return Unc(self.mean_value + other.mean_value, self.sigma_low, self.sigma_up)

        rand_self = randn_asym(self.mean_value, self.sigma_low, self.sigma_up,
                               random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, other.sigma_low, other.sigma_up,
                                random_seed=other.seed)

        rand_result = rand_self + rand_other

        return self.eval(rand_result)

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

        try:
            if isinstance(other, (int, float)):
                return Unc(self.mean_value + other, self.sigma_low, self.sigma_up)

            raise ValueError("Left-hand operand must be either a built-in numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

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

        try:
            if not isinstance(other, (int, float, Unc)):
                raise ValueError("Right-hand operand must be either a built-in\
                                 numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

        if isinstance(other, (int, float)):
            return Unc(self.mean_value - other, self.sigma_low, self.sigma_up)

        if self.seed == other.seed:
            return Unc(0., 0., 0.)

        if other.is_exact:
            return Unc(self.mean_value - other.mean_value, self.sigma_low, self.sigma_up)

        rand_self = randn_asym(self.mean_value, self.sigma_low, self.sigma_up,
                               random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, other.sigma_low, other.sigma_up,
                                random_seed=other.seed)

        rand_result = rand_self - rand_other

        return self.eval(rand_result)

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

        try:
            if isinstance(other, (int, float)):
                return Unc(self.mean_value - other, self.sigma_low, self.sigma_up)

            raise ValueError("Left-hand operand must be either a built-in numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

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

        try:
            if not isinstance(other, (int, float, Unc)):
                raise ValueError("Right-hand operand must be either a built-in\
                                 numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

        if isinstance(other, (int, float)):
            return Unc(self.mean_value*other, self.sigma_low*other, self.sigma_up*other)

        if other.is_exact:
            return (Unc(self.mean_value*other.mean_value, self.sigma_low*other.mean_value,
                        self.sigma_up*other.mean_value))

        rand_self = randn_asym(self.mean_value, self.sigma_low, self.sigma_up,
                               random_seed=self.seed)
        rand_other = randn_asym(other.mean_value, other.sigma_low, other.sigma_up,
                                random_seed=other.seed)

        rand_result = rand_self*rand_other

        return self.eval(rand_result)

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

        try:
            if isinstance(other, (int, float)):
                return Unc(self.mean_value*other, self.sigma_low*other, self.sigma_up*other)

            raise ValueError("Left-hand operand must be either a built-in numerical type or Unc")

        except ValueError:
            print("ValueError")
            raise

    def __pow__(self, exponent):
        """Calculate self**exponent

        Parameters
        ----------
        self : Unc
        exponent : float

        Returns
        -------
        self**exponent : Unc
        """

        if self.is_exact:
            return Unc(self.mean_value**exponent, 0., 0.)

        rand_self = randn_asym(self.mean_value, self.sigma_low, self.sigma_up,
                               random_seed=self.seed)

        rand_result = rand_self**exponent

        return self.eval(rand_result)
