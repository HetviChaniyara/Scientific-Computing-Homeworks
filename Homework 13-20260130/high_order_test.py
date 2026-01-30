import math
import numpy as np
import high_order_template as program

"""
 Test limiters
"""


def test_minmod2():
    assert np.isclose(program.minmod2(+1.0, +2.0), +1.0)
    assert np.isclose(program.minmod2(+2.0, +1.0), +1.0)
    assert np.isclose(program.minmod2(-1.0, -2.0), -1.0)
    assert np.isclose(program.minmod2(-2.0, -1.0), -1.0)

    assert np.isclose(program.minmod2(-1.0, +2.0), 0.0)
    assert np.isclose(program.minmod2(+1.0, -2.0), 0.0)


def test_minmod3():
    assert np.isclose(program.minmod3(+1.0, +2.0, +3.0), +1.0)
    assert np.isclose(program.minmod3(+2.0, +3.0, +1.0), +1.0)
    assert np.isclose(program.minmod3(+3.0, +1.0, +2.0), +1.0)

    assert np.isclose(program.minmod3(-1.0, -2.0, -3.0), -1.0)
    assert np.isclose(program.minmod3(-2.0, -3.0, -1.0), -1.0)
    assert np.isclose(program.minmod3(-3.0, -1.0, -2.0), -1.0)

    assert np.isclose(program.minmod3(-1.0, +2.0, +3.0), 0.0)
    assert np.isclose(program.minmod3(+2.0, +3.0, -1.0), 0.0)
    assert np.isclose(program.minmod3(+3.0, -1.0, +2.0), 0.0)

    assert np.isclose(program.minmod3(+1.0, -2.0, -3.0), 0.0)
    assert np.isclose(program.minmod3(-2.0, -3.0, +1.0), 0.0)
    assert np.isclose(program.minmod3(-3.0, +1.0, -2.0), 0.0)


def test_slope_limiters():
    values_const = np.array([1, 1, 1])
    values_lin = np.array([0.5, 1.0, 1.5])
    values_quad = np.array([0.25, 0.3, 2.25])
    dx = 0.5

    assert np.isclose(program.slope_limiters("const", values_const, dx), 0.0)
    assert np.isclose(program.slope_limiters("const", values_lin, dx), 0.0)
    assert np.isclose(program.slope_limiters("const", values_quad, dx), 0.0)

    assert np.isclose(program.slope_limiters("nolim", values_const, dx), 0.0)
    assert np.isclose(program.slope_limiters("nolim", values_lin, dx), 1.0)
    assert np.isclose(program.slope_limiters("nolim", values_quad, dx), 2.0)

    assert np.isclose(program.slope_limiters("minmod", values_const, dx), 0.0)
    assert np.isclose(program.slope_limiters("minmod", values_lin, dx), 1.0)
    assert np.isclose(program.slope_limiters("minmod", values_quad, dx), 0.1)

    assert np.isclose(program.slope_limiters("MC", values_const, dx), 0.0)
    assert np.isclose(program.slope_limiters("MC", values_lin, dx), 1.0)
    assert np.isclose(program.slope_limiters("MC", values_quad, dx), 0.2)


"""
 Test Heun with Dahlquist test problem
"""


def test_advance_in_time_heun():

    dts = [1/2**c for c in range(0, 6)]
    def fu_rhs(x): return -x
    def fu_exact(t): return math.exp(-t)

    solution_computed = [program.advance_in_time_heun(
        1.0, fu_rhs, dt) - fu_exact(dt) for dt in dts]
    solution_expected = [0.13212055882855767, 0.018469340287366576, 0.0024492169285951215,
                         0.0003155974154045449, 4.0062186524192356e-05, 5.04677365587014e-06]

    assert all(np.isclose(a, b)
               for a, b in zip(solution_computed, solution_expected))


def main():
    test_minmod2()
    test_minmod3()
    test_slope_limiters()
    test_advance_in_time_heun()


if __name__ == '__main__':
    main()
