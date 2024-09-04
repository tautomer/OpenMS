import sys
from typing import Callable, Union

import numpy as np
from scipy.integrate import RK45


def rk4(func: Callable, t0: float, y0: Union[float, np.array], dt: float):
    """Runge-Kutta 4 integrator for an arbitrary univariate function.

    :param func: derivate of the function needs to be integrated. dy/dt
    :type func: Callable
    :param t0: initial time of current step
    :type t0: float
    :param y0: value of the function y at current step
    :type y0: Union[float, numpy.array]
    :param dt: time step
    :type dt: float
    :return: value of function y at t0 + dt
    :rtype: Union[float, numpy.array]
    """
    k1 = func(t0, y0)
    k2 = func(t0 + dt / 2.0, y0 + k1 * dt / 2.0)
    k3 = func(t0 + dt / 2.0, y0 + k2 * dt / 2.0)
    k4 = func(t0 + dt, y0 + k3 * dt)
    return y0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0


def rk45(func: Callable, t0: float, y0: Union[float, np.array], dt: float):
    """Dormand–Prince (a Runge-Kutta 45) integrator for an arbitrary univariate
    function. The coefficients are taken from the original paper
    https://www.sciencedirect.com/science/article/pii/0771050X80900133

    :param func: derivate of the function needs to be integrated. dy/dt
    :type func: Callable
    :param t0: initial time of current step
    :type t0: float
    :param y0: value of the function y at current step
    :type y0: Union[float, numpy.array]
    :param dt: time step
    :type dt: float
    :return: value of function y at t0 + dt
    :rtype: Union[float, numpy.array]
    """

    k1 = dt * func(t0, y0)
    k2 = dt * func(t0 + 1 / 5 * dt, y0 + 1 / 5 * k1)
    k3 = dt * func(t0 + 3 / 10 * dt, y0 + 3 / 40 * k1 + 9 / 40 * k2)
    k4 = dt * func(t0 + 4 / 5 * dt, y0 + 44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3)
    k5 = dt * func(
        t0 + 8 / 9 * dt,
        y0 + 19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4,
    )
    k6 = dt * func(
        t0 + dt,
        y0
        + 9017 / 3168 * k1
        - 355 / 33 * k2
        + 46732 / 5247 * k3
        + 49 / 176 * k4
        - 5103 / 18656 * k5,
    )
    return (
        y0
        + 35 / 384 * k1
        + 500 / 1113 * k3
        + 125 / 192 * k4
        - 2187 / 6784 * k5
        + 11 / 84 * k6
    )


def rk45(
    func: Callable,
    t0: float,
    y0: Union[float, np.array],
    dt: float,
    tol=5e-4,
    max_iter=50,
):
    """Dormand–Prince (a Runge-Kutta 45) integrator for an arbitrary univariate
    function. The coefficients are taken from the original paper
    https://www.sciencedirect.com/science/article/pii/0771050X80900133

    :param func: derivate of the function needs to be integrated. dy/dt
    :type func: Callable
    :param t0: initial time of current step
    :type t0: float
    :param y0: value of the function y at current step
    :type y0: Union[float, numpy.array]
    :param dt: time step
    :type dt: float
    :return: value of function y at t0 + dt
    :rtype: Union[float, numpy.array]
    """
    t = t0
    t_end = t0 + dt
    count = 0
    while t < t_end and count < max_iter:
        if t + dt > t_end:
            dt = t_end - t
        y0, t, dt = rk45_one_step(func, t, y0, dt, tol)
        count += 1
        # print(dt, t)

    if count >= max_iter:
        y0, _, _ = rk45_one_step(func, t, y0, t_end - t, tol, check_error=False)

    # print("Done RK4")
    return y0


def rk45_one_step(
    func: Callable,
    t0: float,
    y0: Union[float, np.array],
    dt: float,
    tol,
    check_error=True,
):
    k1 = dt * func(t0, y0)
    k2 = dt * func(t0 + 1 / 5 * dt, y0 + 1 / 5 * k1)
    k3 = dt * func(t0 + 3 / 10 * dt, y0 + 3 / 40 * k1 + 9 / 40 * k2)
    k4 = dt * func(t0 + 4 / 5 * dt, y0 + 44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3)
    k5 = dt * func(
        t0 + 8 / 9 * dt,
        y0 + 19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4,
    )
    k6 = dt * func(
        t0 + dt,
        y0
        + 9017 / 3168 * k1
        - 355 / 33 * k2
        + 46732 / 5247 * k3
        + 49 / 176 * k4
        - 5103 / 18656 * k5,
    )
    k7 = dt * func(
        t0 + dt,
        y0
        + 35 / 384 * k1
        + 500 / 1113 * k3
        + 125 / 192 * k4
        - 2187 / 6784 * k5
        + 11 / 84 * k6,
    )
    y_next = (
        y0
        + 35 / 384 * k1
        + 500 / 1113 * k3
        + 125 / 192 * k4
        - 2187 / 6784 * k5
        + 11 / 84 * k6
    )
    y_error = (
        (35 / 384 - 5179 / 57600) * k1
        + (500 / 1113 - 7571 / 16695) * k3
        + (125 / 192 - 393 / 640) * k4
        + (-2187 / 6784 + 92097 / 339200) * k5
        + (11 / 84 - 187 / 2100) * k6
        + (-1 / 40) * k7
    )

    if check_error:
        error_norm = np.linalg.norm(y_error) / np.linalg.norm(y_next) * dt
        if error_norm <= tol:
            t0 = t0 + dt
            y0 = y_next
        dt *= min(max(0.2, 0.8 * (tol / error_norm) ** (1 / 5)), 5.0)
    else:
        t0 = t0 + dt
        y0 = y_next

    return y0, t0, dt


def scipy_integrator(f: callable, t0: float, y: float, dt: float):
    scipy_rk4 = RK45(f, t0, y, t0 + dt, max_step=dt)
    count = 0
    while scipy_rk4.status == "running":
        scipy_rk4.step()
        count += 1
    # print(f"{count} iterations")
    return scipy_rk4.y


# class RungeKutta45:
#     """A NumPy version of the Dormand–Prince method. Note that due to small sizes of the
#     C, A, B, and k arrays, it is 2x slower than the hardcoded version above.
#     """
#
#     # values and notations of the Butcher table is taken from
#     # https://www.sciencedirect.com/science/article/pii/0771050X80900133
#     C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
#     A = np.array(
#         [
#             [0, 0, 0, 0, 0],
#             [1 / 5, 0, 0, 0, 0],
#             [3 / 40, 9 / 40, 0, 0, 0],
#             [44 / 45, -56 / 15, 32 / 9, 0, 0],
#             [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
#             [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
#         ]
#     )
#     B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]).reshape(
#         -1, 1
#     )
#     # 6 derivatives used in the integrator
#     k = np.empty((6, 3))
#
#     def __call__(
#         self, func: Callable, t0: float, y0: Union[float, np.array], dt: float
#     ):
#         """Dormand–Prince (a Runge-Kutta 45) integrator for an arbitrary univariate
#         function.
#
#         :param func: derivate of the function needs to be integrated. dy/dt
#         :type func: Callable
#         :param t0: initial time of current step
#         :type t0: float
#         :param y0: value of the function y at current step
#         :type y0: Union[float, numpy.array]
#         :param dt: time step
#         :type dt: float
#         :return: value of function y at t0 + dt
#         :rtype: Union[float, numpy.array]
#         """
#         for i, (c, a) in enumerate(zip(self.C, self.A)):
#             dy = np.dot(a[:i], self.k[:i]) * dt
#             self.k[i] = func(t0 + c * dt, y0 + dy)
#         return y0 + np.dot(self.k.T, self.B)[0] * dt
