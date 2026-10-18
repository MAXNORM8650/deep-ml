def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    m, n = len(g_coeffs), len(h_coeffs)
    grad_g = g_coeffs_at_x = 0
    grad_h = h_coeffs_at_x = 0

    for i in range(m):
        g_coeffs_at_x += g_coeffs[i] * x**(m - 1 - i)

    for i in range(m - 1):
        grad_g += g_coeffs[i] * (m - 1 - i) * x**(m - 2 - i)

    for i in range(n):
        h_coeffs_at_x += h_coeffs[i] * x**(n - 1 - i)

    for i in range(n - 1):
        grad_h += h_coeffs[i] * (n - 1 - i) * x**(n - 2 - i)

    if h_coeffs_at_x == 0:
        raise ZeroDivisionError("h(x) cannot be zero")

    return (grad_g * h_coeffs_at_x - g_coeffs_at_x * grad_h) / h_coeffs_at_x**2