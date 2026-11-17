import math

# ---------------------------------------------------------------------------
# Spherical Bessel functions via upward recurrence on plain floats.
#
# The recurrence
#
#     f_{n+1}(x) = (2n + 1)/x * f_n(x) - f_{n-1}(x)
#
# is used for both j_n and y_n.  It is numerically stable in the upward
# direction for j_n when x is not too small, and for y_n it is stable for
# x larger than roughly n.  For very small x the functions underflow or
# overflow, which is the documented awkward edge of this library.
# ---------------------------------------------------------------------------


def _spherical_jn_series(n: int, x: float) -> float:
    """Direct series for j_n(x) valid for small x and small n.

    j_n(x) = x^n / (2n+1)!! * (1 - x^2/(2(2n+3)) + ...)
    We use the leading term only when the correction is below double
    precision, which avoids catastrophic cancellation in the recurrence
    for tiny x.
    """
    if x == 0.0:
        return 1.0 if n == 0 else 0.0
    # Leading term: x^n / (2n+1)!!
    # (2n+1)!! = (2n+1) * (2n-1) * ... * 3 * 1
    double_factorial = 1.0
    for k in range(1, 2 * n + 2, 2):
        double_factorial *= k
    leading = (x ** n) / double_factorial
    # First correction term: 1 - x^2 / (2*(2n+3))
    correction = 1.0 - (x * x) / (2.0 * (2 * n + 3))
    return leading * correction


def spherical_jn(n: int, x: float) -> float:
    """Return the spherical Bessel function of the first kind j_n(x).

    Parameters
    ----------
    n : int
        Non-negative order.
    x : float
        Real argument.

    Returns
    -------
    float
        j_n(x).  For very small x the result may underflow to 0.0;
        for very large n and x the recurrence can overflow to inf.

    Raises
    ------
    ValueError
        If n is negative.
    """
    if n < 0:
        raise ValueError("order n must be non-negative")
    if x < 0.0:
        # j_n(-x) = (-1)^n j_n(x)
        sign = -1.0 if (n % 2) else 1.0
        return sign * spherical_jn(n, -x)
    if x == 0.0:
        return 1.0 if n == 0 else 0.0

    # For small x and modest n the series is accurate and avoids the
    # cancellation that the upward recurrence suffers near x=0.
    if x < 1e-8 and n <= 10:
        return _spherical_jn_series(n, x)

    # Seed with j_0 and j_1, then recur upward.
    j0 = math.sin(x) / x
    if n == 0:
        return j0
    j1 = (math.sin(x) / (x * x)) - (math.cos(x) / x)
    if n == 1:
        return j1

    jm = j0
    j = j1
    for k in range(1, n):
        jp = (2 * k + 1) / x * j - jm
        jm = j
        j = jp
    return j


def spherical_yn(n: int, x: float) -> float:
    """Return the spherical Bessel function of the second kind y_n(x).

    Parameters
    ----------
    n : int
        Non-negative order.
    x : float
        Real argument, must be positive.

    Returns
    -------
    float
        y_n(x).  For very small x the result may overflow to inf.

    Raises
    ------
    ValueError
        If n is negative or x <= 0.
    """
    if n < 0:
        raise ValueError("order n must be non-negative")
    if x <= 0.0:
        raise ValueError("x must be positive for y_n")

    # Seed with y_0 and y_1, then recur upward.
    y0 = -math.cos(x) / x
    if n == 0:
        return y0
    y1 = -math.cos(x) / (x * x) - math.sin(x) / x
    if n == 1:
        return y1

    ym = y0
    y = y1
    for k in range(1, n):
        yp = (2 * k + 1) / x * y - ym
        ym = y
        y = yp
    return y
