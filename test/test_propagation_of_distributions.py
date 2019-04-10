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

import pytest

from numpy import array, array_equal

from asym_uncertainty import Unc

class TestPropagation(object):
    def test_n_random_mismatch(self):
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 3.]), store=True)
        b = Unc(1., 0.5, 0.5, random_values=array([2., 1., 2., 1.]), store=True)

        with pytest.warns(UserWarning) as record:
            add = a+b
            sub = a-b
            mul = a*b
            div = a/b
            power = a**b

        assert len(record) == 5
        assert "mismatch" in record[0].message.args[0]

        assert len(add.random_values) == 3
        assert array_equal(array([3., 3., 5.]), add.random_values)
        assert len(sub.random_values) == 3
        assert array_equal(array([-1., 1., 1.]), sub.random_values)
        assert len(mul.random_values) == 3
        assert array_equal(array([2., 2., 6.]), mul.random_values)
        assert len(div.random_values) == 3
        assert array_equal(array([0.5, 2.]), div.random_values[0:2])
        assert len(power.random_values) == 3
        assert array_equal(array([1., 2., 9.]), power.random_values)

    def test_addition(self):
        # Test without storage of random values
        with pytest.warns(UserWarning):
            a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]))
        b = 1.+a
        assert len(b.random_values) == 1

        # Test with storage of random values
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = 1.+a
        assert len(b.random_values) == 3
        assert array_equal(array([2., 3., 5.]), b.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = a + 1.
        assert len(b.random_values) == 3
        assert array_equal(array([2., 3., 5.]), b.random_values)

        # Test transitive relation, i.e. if b = f(a), and c = f(b), then the set of random
        # values should be transmitted from a to c
        c = 1.+b
        assert len(c.random_values) == 3
        assert array_equal(array([3., 4., 6.]), c.random_values)

        # Check that store=False dominates
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        with pytest.warns(UserWarning):
            b = Unc(1., 0.5, 0.5, random_values=array([1., 1., 2.]))

        c = b + a
        assert len(c.random_values) == 3
        assert not array_equal(array([2., 3., 6.]), c.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = Unc(1., 0.5, 0.5, random_values=array([1., 1., 2.]), store=True)

        c = b + a
        assert len(c.random_values) == 3
        assert array_equal(array([2., 3., 6.]), c.random_values)

    def test_division(self):
        # Test without storage of random values
        with pytest.warns(UserWarning):
            a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]))
        b = 1./a
        assert len(b.random_values) == 1

        # Test with storage of random values
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = 1./a
        assert len(b.random_values) == 3
        assert array_equal(array([1., 0.5, 0.25]), b.random_values)

        # Test transitive relation, i.e. if b = f(a), and c = f(b), then the set of random
        # values should be transmitted from a to c
        c = 1./b
        assert len(c.random_values) == 3
        assert array_equal(array([1., 2., 4.]), c.random_values)

        # Check that store=False dominates
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        with pytest.warns(UserWarning):
            b = Unc(1., 0.5, 0.5, random_values=array([1., 1., 2.]))

        c = b/a
        assert len(c.random_values) == 3
        assert not array_equal(array([1., 0.5, 0.5]), c.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = Unc(1., 0.5, 0.5, random_values=array([1., 1., 2.]), store=True)

        c = b/a
        assert len(c.random_values) == 3
        assert array_equal(array([1., 0.5, 0.5]), c.random_values)

    def test_power(self):
        # Test without storage of random values
        with pytest.warns(UserWarning):
            a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 2.]))
        b = 2.**a
        assert len(b.random_values) == 1

        # Test with storage of random values
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = 2.**a
        assert len(b.random_values) == 3
        assert array_equal(array([2., 4., 16.]), b.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = a**2.
        assert len(b.random_values) == 3
        assert array_equal(array([1., 4., 16.]), b.random_values)

        # Test transitive relation, i.e. if b = f(a), and c = f(b), then the set of random
        # values should be transmitted from a to c
        c = 2.**b
        assert len(c.random_values) == 3
        assert array_equal(array([2., 16., 65536.]), c.random_values)

        # Check that store=False dominates
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        with pytest.warns(UserWarning):
            b = Unc(1., 0.5, 0.5, random_values=array([1., 2., 2.]))

        c = b**a
        assert len(c.random_values) == 3
        assert not array_equal(array([1., 4., 16.]), c.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = Unc(1., 0.5, 0.5, random_values=array([1., 1., 2.]), store=True)

        c = a**b
        assert len(c.random_values) == 3
        assert array_equal(array([1., 2., 16.]), c.random_values)

        a = Unc(2., 0., 0., n_random=3)
        b = Unc(1., 0.5, 0.5, random_values=array([1., 2., 3.]), store=True)

        c = a**b
        assert len(c.random_values) == 3
        assert array_equal(array([2., 4., 8.]), c.random_values)

    def test_subtraction(self):
        # Test without storage of random values
        with pytest.warns(UserWarning):
            a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]))
        b = 1.-a
        assert len(b.random_values) == 1

        # Test with storage of random values
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = 1.-a
        assert len(b.random_values) == 3
        assert array_equal(array([0., -1., -3.]), b.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = a - 1.
        assert len(b.random_values) == 3
        assert array_equal(array([0., 1., 3.]), b.random_values)

        # Test transitive relation, i.e. if b = f(a), and c = f(b), then the set of random
        # values should be transmitted from a to c
        c = 1.-b
        assert len(c.random_values) == 3
        assert array_equal(array([1., 0., -2.]), c.random_values)

        # Check that store=False dominates
        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        with pytest.warns(UserWarning):
            b = Unc(1., 0.5, 0.5, random_values=array([1., 2., 2.]))

        c = b - a
        assert len(c.random_values) == 3
        assert not array_equal(array([0., 0., -2.]), c.random_values)

        a = Unc(1., 0.5, 0.5, random_values=array([1., 2., 4.]), store=True)
        b = Unc(1., 0.5, 0.5, random_values=array([1., 2., 2.]), store=True)

        c = b - a
        assert len(c.random_values) == 3
        assert array_equal(array([0., 0., -2.]), c.random_values)

