""" Auxiliary functions for Unc, for example analytical expressions of distributions"""

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

from numpy import exp, pi, sqrt
from scipy.stats import norm

def auxiliary_a(z, sigma_num, sigma_denom):
    """ Auxiliary function a for the ratio of two normal distributions """
    return sqrt(z*z/(sigma_num*sigma_num) + 1./(sigma_denom*sigma_denom))

def auxiliary_b(z, mu_num, sigma_num, mu_denom, sigma_denom):
    """ Auxiliary function b for the ratio of two normal distributions """
    return z*mu_num/(sigma_num*sigma_num) + mu_denom/(sigma_denom*sigma_denom)

def auxiliary_c(mu_num, sigma_num, mu_denom, sigma_denom):
    """ Auxiliary function c for the ratio of two normal distributions """
    return mu_num*mu_num/(sigma_num*sigma_num) + mu_denom*mu_denom/(sigma_denom*sigma_denom)

def auxiliary_d(z, mu_num, sigma_num, mu_denom, sigma_denom):
    """ Auxiliary function d for the ratio of two normal distributions """
    return exp((auxiliary_b(z, mu_num, sigma_num, mu_denom, sigma_denom)**2-
                auxiliary_c(mu_num, sigma_num, mu_denom, sigma_denom)*
                auxiliary_a(z, sigma_num, sigma_denom)**2)/
               (2.*auxiliary_a(z, sigma_num, sigma_denom)**2)
              )

def gaussian_ratio_pdf(z, mu_num, sigma_num, mu_denom, sigma_denom):
    """ General ratio of two normal distributions

    This function calculates the probability density function PDF for the ratio z=x/y of two
    normal-distributed random variables x and y, where PDF(x) = N(mu_x, sigma_x) and
    PDF(y) = N(mu_y, sigma_y). The mean values mu and sigma can have arbitrary values.

    Parameters
    ----------
    mu_num: float
        Mean value of the normal distribution for the numerator
    sigma_num: positive float
        Standard deviation of the normal distribution for the numerator
    mu_denom: float
        Mean value of the normal distribution for the denominator
    sigma_denom: positive float
        Standard deviation of the normal distribution for the denominator

    Returns
    -------
    float
        PDF(z)
    """

    return (auxiliary_b(z, mu_num, sigma_num, mu_denom, sigma_denom)*
            auxiliary_d(z, mu_num, sigma_num, mu_denom, sigma_denom)/
            (auxiliary_a(z, sigma_num, sigma_denom)**3)*
            (1./(sqrt(2.*pi)*sigma_num*sigma_denom))*
            (norm.cdf(auxiliary_b(z, mu_num, sigma_num, mu_denom, sigma_denom)/
                      auxiliary_a(z, sigma_num, sigma_denom)) -
             norm.cdf(-auxiliary_b(z, mu_num, sigma_num, mu_denom, sigma_denom)/
                      auxiliary_a(z, sigma_num, sigma_denom))
            ) +
            exp(-auxiliary_c(mu_num, sigma_num, mu_denom, sigma_denom)*0.5)/
            (auxiliary_a(z, sigma_num, sigma_denom)**2*pi*sigma_num*sigma_denom)
           )
