"""Get the shortest coverage interval and most probable value of randomly sampled
values from a distribution"""

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

from numpy import argmax, extract, histogram, median
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

from .mc_statistics import cdf, shortest_coverage

def evaluate(rand_result, force_inside_shortest_coverage=True, use_kde=True):
    """Implementation of Unc.eval()"""

    s_cov = shortest_coverage(cdf(rand_result))
    if force_inside_shortest_coverage:
    	
    	if use_kde:
    	    data_inside = extract((rand_result >= s_cov[0])*(rand_result <= s_cov[1]), rand_result)
    	    
    	    plain_kde = gaussian_kde(data_inside)
    	    kde = lambda x: (-1)*plain_kde.evaluate(x)
    	    most_probable = minimize(kde,x0=median(data_inside),
    	        method='TNC',tol=1e-5,bounds=((min(data_inside),max(data_inside)),)).x[0]

    	else:
            hist, bins = histogram(
                extract((rand_result >= s_cov[0])*(rand_result <= s_cov[1]), rand_result),
                bins="sqrt")

            most_probable = bins[argmax(hist)]+0.5*(bins[1]-bins[0])

# In the context of asym_uncertainty, this case will never occur.
# However, it was decided to leave the 'else' statement here as a reminder that
# the force_inside_shortest_coverage option is set.
#    else:
#        hist, bins = histogram(rand_result, bins="sqrt")

    

    return ([most_probable, most_probable - s_cov[0], s_cov[1] - most_probable], rand_result)
