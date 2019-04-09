"""Re-definition of some mathematical functions for use with the
Unc class"""

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

from numpy import exp as nexp

from asym_uncertainty import evaluate, Unc
from mc_statistics import randn_asym

def exp(unc):
    """ Calculate exp(u)

    Parameters
    ----------
    u : Unc

    Returns
    -------
    exp(u) : Unc
    """

    try:
        if not isinstance(unc, Unc):
            raise ValueError("Operand must be of type Unc")

    except ValueError:
        print("ValueError")
        raise

    if unc.is_exact:
        return Unc(nexp(unc.mean_value), 0., 0.)

    rand = randn_asym(unc.mean_value, [unc.sigma_low, unc.sigma_up],
                      limits=unc.limits, random_seed=unc.seed,
                      n_random=unc.n_random)

    rand_result = nexp(rand)

    exp_result = evaluate(rand_result, force_inside_shortest_coverage=True)

    if unc.store:
        return Unc(exp_result[0][0], exp_result[0][1], exp_result[0][2],
                   random_values=exp_result[1])
    return Unc(exp_result[0][0], exp_result[0][1], exp_result[0][2])
