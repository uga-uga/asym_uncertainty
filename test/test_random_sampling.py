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

from matplotlib.pyplot import hist, plot, subplots
from numpy import linspace, pi, sqrt
from scipy.special import kn
from scipy.stats import cauchy, chi2, lognorm, norm

from asym_uncertainty import exp, gaussian_ratio_pdf, Unc

class TestRandomSampling(object):
    def test_addition(self):
        a = Unc(0., 1., 1., store=True)
        b = Unc(0., 1., 1., store=True)

        c = a + b

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -5.
        x_max = 5.
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel("y = (0.0 - 1.0 + 1.0) + (0.0 - 1.0 + 1.0)")

        y_min = 0.
        y_max = 0.5

        ax.set_ylim([y_min, y_max])
        ax.set_ylabel("p(y) = N(0, sqrt(2))")

        ax.plot(x_values, norm.pdf(x_values, loc=0., scale=sqrt(2.)), color='red',
                label="Analytical expression")

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.legend()
        fig.savefig('test/test_addition.pdf')

    def test_subtraction(self):
        a = Unc(0., 1., 1., store=True)
        b = Unc(0., 1., 1., store=True)

        c = a - b

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -5.
        x_max = 5.
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel("y = (2.0 - 0.1 + 0.1) / (2.0 - 0.1 + 0.1)")

        y_min = 0.
        y_max = 0.5

        ax.set_ylim([y_min, y_max])
        ax.set_ylabel("p(y) = N(0, sqrt(2))(y)")

        ax.plot(x_values, norm.pdf(x_values, loc=0., scale=sqrt(2.)), color='red',
                label="Analytical expression")

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.legend()
        fig.savefig('test/test_subtraction.pdf')

    def test_multiplication(self):
        a = Unc(0., 1., 1., store=True)
        b = Unc(0., 1., 1., store=True)

        c = a * b

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -3.
        x_max = 3.
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel("y = (0.0 - 1.0 + 1.0) * (0.0 - 1.0 + 1.0)")

        y_min = 0.
        y_max = 2.5

        ax.set_ylim([y_min, y_max])
        ax.set_ylabel("p(y) = $K_0$(y)/$\pi$")

        ax.plot(x_values, kn(0, x_values)/pi, color='red',
                label="Analytical expression")
        ax.plot(-x_values, kn(0, x_values)/pi, color='red')

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.legend()
        fig.savefig('test/test_multiplication.pdf')

    def test_division(self):
        a = Unc(2., 0.1, 0.1, store=True)
        b = Unc(2., 0.1, 0.1, store=True)

        c = a / b

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = 0.5
        x_max = 1.5
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)

        ax.set_xlim([x_min, x_max])
        ax.set_xlabel("y = (0.0 - 1.0 + 1.0) * (0.0 - 1.0 + 1.0)")

        y_min = 0.
        y_max = 6.
        ax.set_ylim([y_min, y_max])
        ax.set_ylabel("p(y)")

        ax.plot(x_values, gaussian_ratio_pdf(x_values, 2., 0.1, 2., 0.1), color='red',
                label="Analytical expression")

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.legend()
        fig.savefig('test/test_division.pdf')

    def test_power(self):
        a = Unc(0., 1., 1., store=True)

        c = a*a

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -0.1
        x_max = 3.
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)
        ax.set_xlim([x_min, x_max])

        y_min = 0.
        y_max = 2.
        ax.set_ylim([y_min, y_max])

        ax.set_xlabel("y = (0.0 - 1.0 + 1.0) ** 2")
        ax.set_ylabel("p(y) = $\chi^2$(y, 1)")
        ax.plot(x_values, chi2.pdf(x_values, 1), color='red',
                label="Analytical expression")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")
        ax.legend()
        fig.savefig('test/test_power.pdf')

    def test_exp(self):
        a = Unc(0., 1., 1., store=True)

        c = exp(a)

        assert len(c.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = 0.
        x_max = 6.
        n_x = int(sqrt(len(c.random_values)))
        x_values = linspace(x_min, x_max, n_x)
        ax.set_xlim([x_min, x_max])

        y_min = 0.
        y_max = 1.
        ax.set_ylim([y_min, y_max])

        ax.set_xlabel("y = exp(0.0 - 1.0 + 1.0)")
        ax.set_ylabel("p(y) = Log-normal(1, 1)(y)")
        ax.plot(x_values, lognorm.pdf(x_values, 1., 0.), color='red',
                label="Analytical expression")

        ax.plot([c.mean_value, c.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([c.mean_value - c.sigma_low, c.mean_value - c.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([c.mean_value + c.sigma_up, c.mean_value + c.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.hist(c.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")
        ax.legend()
        fig.savefig('test/test_exp.pdf')

    def test_asymmetric(self):
        a = Unc(0., 0.5, 1., store=True)

        assert len(a.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -3.
        x_max = 3.
        n_x = int(sqrt(len(a.random_values)))
        x_values = linspace(x_min, x_max, n_x)
        ax.set_xlim([x_min, x_max])

        y_min = 0.
        y_max = 1.
        ax.set_ylim([y_min, y_max])

        ax.set_xlabel("y = 0.0 - 0.5 + 1.0")
        ax.set_ylabel("p(y)")
        ax.plot(x_values, norm.pdf(x_values, 0., 0.5), color='red',
                label="Analytical expression")
        ax.plot(x_values, norm.pdf(x_values, 0., 1.), color='red')

        ax.plot([a.mean_value, a.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([a.mean_value - a.sigma_low, a.mean_value - a.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([a.mean_value + a.sigma_up, a.mean_value + a.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.hist(a.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")
        ax.legend()
        fig.savefig('test/test_asymmetric.pdf')

    def test_truncated(self):
        a = Unc(0., 1., 1., limits=[-1., 1.], store=True)
        a.update_limits()

        assert len(a.random_values) > 1

        fig, ax = subplots(1,1)
        
        x_min = -2.
        x_max = 2.
        n_x = int(sqrt(len(a.random_values)))
        x_values = linspace(x_min, x_max, n_x)
        ax.set_xlim([x_min, x_max])

        y_min = 0.
        y_max = 1.
        ax.set_ylim([y_min, y_max])

        ax.set_xlabel("y = 0.0 - 1.0 + 1.0")
        ax.set_ylabel("p(y) = N(0, 1)/0.6827, -1 <= y <= 1")
        ax.plot(x_values, norm.pdf(x_values, 0., 1.)/0.6827, color='red',
                label="Analytical expression")

        ax.plot([a.mean_value, a.mean_value], [y_min, y_max], color='black',
                label="Most probable value")
        ax.plot([a.mean_value - a.sigma_low, a.mean_value - a.sigma_low], 
                [y_min, y_max], '--', color='black',
                label="Shortest coverage")
        ax.plot([a.mean_value + a.sigma_up, a.mean_value + a.sigma_up], 
                [y_min, y_max], '--', color='black')

        ax.hist(a.random_values, density=True, bins='sqrt', color='grey',
                label="Monte Carlo")
        ax.legend()
        fig.savefig('test/test_truncated.pdf')
