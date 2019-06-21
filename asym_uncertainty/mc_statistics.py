"""Monte-Carlo method for propagation of asymmetric uncertainties"""

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

from numpy import (absolute, argmax, argmin, array, diff, histogram, inf, insert, linspace,
                   ndarray, roll, shape, size, sort, zeros)
from numpy.random import normal, seed, uniform
from scipy.stats import norm

SC_TOLERANCE = 0.01 # Needed for the estimation of the uncertainty of the shortest coverage interval
SC_UNCERTAINTY_DEFAULT = 0.05 # Relative uncertainty of the shortest
# coverage interval if it cannot be determined by the derivative method

def cdf(rand):
    """Calculates the cumulative distribution function (CDF) for \
    unordered samples x_i from a distribution.
    The more random values, the better the approximation of the real continuous CDF will be.

    Parameters
    ----------
    rand : array_like
        Array of random numbers

    Returns
    -------
    cdf : ndarray
        cdf[0] are the values CDF(x_i)
        cdf[1] are the sorted values x_i
        cdf[x_0] == 0. and cdf[x_N] == 1., where N is the number of random samples

    """
    return [sort(rand.flatten()), linspace(0., 1., size(rand))]

def shortest_coverage(cum_dis_fun, coverage_percent=68.27, uncertainty_estimate=False):
    """Calculates the shortest interval |x1 - x0| that covers coverage_percent of \
            a probability distribution function PDF, \
            i.e. min(|x1 - x0|) : |CDF(x1) - CDF(x0)| == coverage_percent

    Parameters
    ----------
    cum_dis_fun : array_like
        Cumulative distribution function CDF of the PDF as returned by cdf()
    coverage_percent : float
        Coverage interval with a value in the interval (0,100) in percent
    uncertainty_estimate : bool, optional
        Estimate the uncertainty dx1 and dx0 of x1 and x0 that is caused \
        by the discrete sampling of the CDF, i.e. take into account the finite bin width.

    Returns
    -------
    shortest_coverage : ndarray
        [x0, x1], where x0 < x1 or
        [x0, x1, dx0, dx1] if uncertainty_estimate == True

    """
    n_rand = shape(cum_dis_fun)[1]

    coverage_interval = int(coverage_percent*0.01*n_rand)

    dist = absolute(cum_dis_fun[0] - roll(cum_dis_fun[0], coverage_interval))

    s_cov = argmin(dist[coverage_interval:])

    if uncertainty_estimate:
        # Estimate the uncertainty of the shortest coverage interval based
        # on the first derivative of the CDF
        # The first derivative d(coverage_interval)/d(x) indicates how much small
        # variations of the shortest coverage interval influence
        # the covered area of the probability distribution.
        # Define the uncertainty limits as the maximum variation in coverage_percent
        # that is acceptable when x is varied
        # The maximum relative variation of coverage_percent is given by SC_TOLERANCE

        # Calculate the lower and upper derivative if possible
        if s_cov == 0:
            derivative = (absolute((dist[s_cov] - dist[s_cov + 1])/
                                   (cum_dis_fun[0][s_cov + 1] - cum_dis_fun[0][s_cov])))
        elif s_cov == n_rand - coverage_interval - 1:
            derivative = (absolute((dist[s_cov] - dist[s_cov - 1])/
                                   (cum_dis_fun[0][s_cov] - cum_dis_fun[0][s_cov - 1])))
        else:
            derivative_low = (absolute(dist[s_cov] - dist[s_cov - 1])/
                              (cum_dis_fun[0][s_cov] - cum_dis_fun[0][s_cov - 1]))

            derivative_up = (absolute(dist[s_cov + 1] - dist[s_cov])/
                             (cum_dis_fun[0][s_cov + 1] - cum_dis_fun[0][s_cov]))

            derivative = absolute(0.5*(derivative_low + derivative_up))

        if derivative == 0.:
            # Simply return an uncertainty of SC_UNCERTAINTY_DEFAULT*(sc[1] - sc[0])
            if cum_dis_fun[0][s_cov] > cum_dis_fun[0][s_cov + coverage_interval]:
                return array([cum_dis_fun[0][s_cov + coverage_interval],
                              cum_dis_fun[0][s_cov],
                              0.10*absolute(cum_dis_fun[0][s_cov + coverage_interval] -
                                            cum_dis_fun[0][s_cov]),
                              0.10*absolute(cum_dis_fun[0][s_cov + coverage_interval] -
                                            cum_dis_fun[0][s_cov])
                             ])

            return array([cum_dis_fun[0][s_cov],
                          cum_dis_fun[0][s_cov + coverage_interval],
                          0.10*absolute(cum_dis_fun[0][s_cov + coverage_interval] -
                                        cum_dis_fun[0][s_cov]),
                          0.10*absolute(cum_dis_fun[0][s_cov + coverage_interval] -
                                        cum_dis_fun[0][s_cov])
                         ])

        if cum_dis_fun[0][s_cov] > cum_dis_fun[0][s_cov + coverage_interval]:
            return array([cum_dis_fun[0][s_cov + coverage_interval],
                          cum_dis_fun[0][s_cov],
                          SC_TOLERANCE/derivative, SC_TOLERANCE/derivative])

        return array([cum_dis_fun[0][s_cov],
                      cum_dis_fun[0][s_cov + coverage_interval],
                      SC_TOLERANCE/derivative, SC_TOLERANCE/derivative])

# Old calculation of uncertainty which is simply based on the distance of the sampled points
# This estimate can be misleading, because for a very flat CDF,
# increasing the number of points would make the result seem more and more precise,
# although, actually, the position of the shortest coverage interval is very ambiguous
#        return array([cum_dis_fun[0][s_cov], cum_dis_fun[0][s_cov + coverage_interval],
#                      cum_dis_fun[0][s_cov + 1] -
#                      cum_dis_fun[0][s_cov], cum_dis_fun[0][s_cov + coverage_interval] -
#                      cum_dis_fun[0][s_cov + coverage_interval - 1]])

    return array([cum_dis_fun[0][s_cov], cum_dis_fun[0][s_cov + coverage_interval]])

def chi2(data, uncertainties, fit, degrees_of_freedom=1):
    """Calculates the (reduced) chi square of theoretical values 'fit' \
    fitted to experimental data 'data' which have uncertainties 'uncertainties'.

    Parameters:
    -----------
        data : ndarray
            Experimental data
        uncertainties : ndarray
            Their uncertainties
        fit : ndarray
            Theoretical values that aim to describe the data
        degrees_of_freedom : int, optional
            Degrees of freedom of the theory

    Returns
    -------

        chi2 : float
            Value of the (reduced) chi2
    """
    return (1./degrees_of_freedom)*ndarray.sum((data - fit)**2/uncertainties**2)

def randn_asym(mean_value, sigma, limits=None, conserve_mean_value=False,
               random_seed=None, n_random=int(1e6)):
    """Create an array of random numbers from a generalized normal distribution \
    that may be asymmetric or truncated.
    Asymmetric here means that left of the maximum mean_value, \
    the distribution has another standard deviation than on the right, \
    i.e. it is made up of a piecewise combination of two normal distributions.
    The sampled values will retain the mean value mean_value if both sides \
    are weighted correctly, which can be controlled via the conserve_mean_value option.
    If conserve_mean_value is set to False, 50% of the values will be on the left \
    and 50% on the right of the maximum for large numbers.
    Truncated means that the sampling does not give negative numbers.
    In this case, the mean value can not be mean_value any more, of course.

    Parameters
    ----------
    mean_value : float
        Mean value of the distribution
    sigma : [float, float]
        Left- and right-hand standard deviation
    limits : [float, float]
        Set lower and upper limit for the random number generator.
        No limit is indicated by entering limit==None or
        limit == [-math.inf, math.inf] using the standard math package
        Default: None
    conserve_mean_value : bool, optional
        Ensures that the mean value of the sampled numbers will be at mean_value.
        Cannot be activated at the same time as force_positive.
        Do not use this option if randn is going to be used in further calculations.
    random_seed : positive int
        Set the seed of numpy's random number generator
    n_random : positive int
        Determines how many random numbers should be generated
        (default: 1e6)

    Returns
    -------
    randn : ndarray
        Array of random numbers
    """

    try:
        if conserve_mean_value:
            lim = sigma[1]/(sigma[0] + sigma[1])
            if limits is not None:
                raise RuntimeWarning("Truncation of negative numbers \
                      and conservation of the mean value at the same time makes no sense.")
        else:
            lim = 0.5
    except RuntimeWarning:
        print("RuntimeWarning")
        raise

    if limits is None:
        limits = [-inf, inf]

    check_num_array_argument(limits, 2, argument_name="Limits", is_increasing=True)
    check_num_array_argument(sigma, 2, argument_name="Sigma", is_positive=True)

    try:
        if random_seed is not None:
            if isinstance(random_seed, int) and random_seed >= 0:
                seed(random_seed)
            else:
                raise ValueError("Random number seed must be positive integer")
    except ValueError:
        print("ValueError")
        raise

    rand = zeros(n_random)
    plusminus = uniform(size=n_random)

    if limits[0] == -inf and limits[1] == inf:
        plus = (plusminus >= lim)*1.
        minus = (plusminus < lim)*1.

        rand += (mean_value + absolute(normal(size=n_random))*sigma[1])*plus
        rand += (mean_value - absolute(normal(size=n_random))*sigma[0])*minus

        if n_random == 1:
            return rand[0]
        else:
            return rand

    if mean_value <= limits[0]:
        y_min = norm.cdf(limits[0], loc=mean_value, scale=sigma[1])
        y_max = norm.cdf(limits[1], loc=mean_value, scale=sigma[1])

        rand = norm.ppf(uniform(y_min, y_max, size=n_random), loc=mean_value, scale=sigma[1])
        if n_random == 1:
            return rand[0]
        else:
            return rand

    if mean_value >= limits[1]:
        y_min = norm.cdf(limits[0], loc=mean_value, scale=sigma[0])
        y_max = norm.cdf(limits[1], loc=mean_value, scale=sigma[0])

        rand = norm.ppf(uniform(y_min, y_max, size=n_random), loc=mean_value, scale=sigma[0])
        if n_random == 1:
            return rand[0]
        else:
            return rand

    # Relative amount of numbers that will be sampled from
    # a normal distribution with sigma[0]
    weight_low = (norm.cdf(mean_value, loc=mean_value, scale=sigma[0]) -
                  norm.cdf(limits[0], loc=mean_value, scale=sigma[0]))
    weight_up = (norm.cdf(limits[1], loc=mean_value, scale=sigma[1]) -
                 norm.cdf(mean_value, loc=mean_value, scale=sigma[1]))
    lim = weight_low/(weight_low + weight_up)

    plus = (plusminus >= lim)*1.
    minus = (plusminus < lim)*1.

    y_min = norm.cdf(limits[0], loc=mean_value, scale=sigma[0])
    y_max = norm.cdf(limits[1], loc=mean_value, scale=sigma[1])

    rand += (norm.ppf(uniform(y_min, 0.5, size=n_random),
                      loc=mean_value, scale=sigma[0])*minus)
    rand += (norm.ppf(uniform(0.5, y_max, size=n_random),
                      loc=mean_value, scale=sigma[1])*plus)

    if n_random == 1:
        return rand[0]
    else:
        return rand

def check_num_array_argument(input_array, array_length, argument_name="Input",
                             is_positive=False, is_increasing=False):
    """ Check whether a given array only contains array_length
    int or float numbers.
    If not, raise a ValueError

    Parameters
    ----------
    input_array : array_like
        Array containing, in principle, anything
    array_length : int
        Desired length of the array
    argument_name : string
        Name of the array to be printed in the error message
    is_positive : bool
        Require the array to contain only positive numbers
    is_increasing : bool
        Require the array to contain strictly increasing elements

    Returns
    -------
    Nothing

    """

    try:
        if len(input_array) != array_length:
            raise ValueError(argument_name, "must be given as an array of", array_length,
                             "float (int) numbers")
        for i in input_array:
            if not isinstance(i, (int, float)):
                raise ValueError(argument_name, "must be given as an array of",
                                 array_length, "float (int) numbers")
        if is_positive:
            for i in input_array:
                if i < 0:
                    raise ValueError(argument_name, "must be an array positive float (int) numbers")
        if is_increasing and not diff(input_array) >= 0.:
            raise ValueError(argument_name, "must be strictly increasing.")
    except ValueError:
        print("ValueError")
        raise
