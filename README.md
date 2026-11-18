# spherical-bessel-functions-kit

Evaluates spherical Bessel functions of the first and second kind for real arguments using upward recurrences on plain Python floats.

```python
from spherical_bessel_functions_kit import spherical_jn, spherical_yn

print(spherical_jn(0, 1.0))   # sin(1)/1
print(spherical_jn(2, 2.0))   # (3/x^3 - 1/x) sin x - 3 cos x / x^2
print(spherical_yn(1, 1.5))   # -cos(x)/x^2 - sin(x)/x
```

## Why this exists

Spherical Bessel functions appear in wave scattering, quantum mechanics, and heat conduction in spheres.  Implementing them without pulling in SciPy is useful when a project must stay dependency-free.  The trade-off here is simplicity: the upward recurrence is accurate for moderate orders and arguments, but it is not a general-purpose replacement for a library that uses continued fractions or asymptotic expansions for large orders.  If you need high accuracy for very large `n` or very small `x`, this is not the right tool.

## The awkward edge

For very small `x` the recurrence for `j_n` suffers cancellation, so this library falls back to a direct series for `x < 1e-8` and `n <= 10`.  For `y_n` and very small `x` the values overflow to infinity — that is a property of the functions, not a bug in the recurrence.  Negative `x` is supported for `j_n` via the parity relation `j_n(-x) = (-1)^n j_n(x)`; `y_n` requires `x > 0`.

## API

- `spherical_jn(n, x)` — spherical Bessel function of the first kind.  Raises `ValueError` if `n < 0`.
- `spherical_yn(n, x)` — spherical Bessel function of the second kind.  Raises `ValueError` if `n < 0` or `x <= 0`.
