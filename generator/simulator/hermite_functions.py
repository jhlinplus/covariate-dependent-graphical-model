"""
The first part of this script contains the functionality for hermite_functions
and is largely copied from
https://github.com/Rob217/hermite-functions/blob/master/hermite_functions/hermite_functions.py
- move_axes=() has been removed since it's not super relevant

The second part of the this script contains the functionality for hermite_polynomials
JL wrote it on July 4, 2024
"""
import numpy as np
from scipy.special import hermite, eval_hermite, factorial

def hermite_functions(n, x, all_n=True, method="recursive"):
    """
    Calculate the Hermite functions up to the nth order at position x, psi_n(x).

    For details see:
    https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions

    If all_n == True, then return all Hermite functions up to n
    If all_n == False, only return nth Hermite function
    If using recursive method, then the latter is more memory efficient as it
    only stores psi_n, psi_{n-1}, and psi_{n-2}

    Uses one of three possible calculation methods:
        'recursive' - Uses recursive method. Most efficient for n > 5.
        'direct'    - Calculates directly using Hermite polynomials.
                      Inefficient due to factorial and Hermite polynomial,
                      although useful for comparison when testing
        'analytic'  - Uses analytic expressions (only for n <= 5)

    Recursion relation:
        psi_n(x) = sqrt(2/n) * x * psi_{n-1}(x) - sqrt((n-1)/n) * psi_{n-2}(x)

    Examples:

    >>> x = np.mgrid[-2:3, 0:4]
    >>> x.shape
    (2, 5, 4)
    >>> n = 5
    >>> psi = hermite_functions(n, x, all_n=False)
    >>> psi.shape
    (2, 5, 4)
    >>> psi = hermite_functions(n, x, all_n=True)
    >>> psi.shape
    (6, 2, 5, 4)
    """

    if method not in ["recursive", "analytic", "direct"]:
        raise ValueError("Method not recognized.")
    if not (issubclass(type(n), int) or issubclass(type(n), np.integer)):
        raise TypeError("n must be an integer.")
    if n < 0:
        raise ValueError("n must be non-negative.")
    if method == "analytic" and (n > 5):
        raise ValueError("n must not be greater than 5 for analytic calculation.")

    if all_n:
        psi_n = _Hermite_all_n(n, x, method)
    else:
        psi_n = _Hermite_single_n(n, x, method)

    return psi_n


def _Hermite_single_n(n, x, method):
    """
    Calculates psi_n(x) for a single value of n.
    """

    if method == "analytic":
        return _H_analytic(n, x)

    if method == "direct":
        return _H_direct(n, x)

    psi_m_minus_2 = _H_analytic(0, x)
    if n == 0:
        return psi_m_minus_2

    psi_m_minus_1 = _H_analytic(1, x)
    if n == 1:
        return psi_m_minus_1

    for m in range(2, n + 1):
        psi_m = _H_recursive(m, x, psi_m_minus_2, psi_m_minus_1)
        psi_m_minus_2 = psi_m_minus_1
        psi_m_minus_1 = psi_m

    return psi_m


def _Hermite_all_n(n, x, method):
    """
    Calcualtes psi_m(x) for all 0 <= m <= n.
    """
    try:
        psi_n = np.zeros((n + 1,) + x.shape)
    except AttributeError:  # x does not have property 'shape'
        psi_n = np.zeros((n + 1, 1))

    if method == "analytic":
        for m in range(n + 1):
            psi_n[m, :] = _H_analytic(m, x)
        return psi_n

    if method == "direct":
        for m in range(n + 1):
            psi_n[m] = _H_direct(m, x)
        return psi_n

    psi_n[0] = _H_analytic(0, x)
    if n == 0:
        return psi_n

    psi_n[1] = _H_analytic(1, x)
    if n == 1:
        return psi_n

    for m in range(2, n + 1):
        psi_n[m] = _H_recursive(m, x, psi_n[m - 2], psi_n[m - 1])

    return psi_n


def _H_recursive(m, x, psi_m_minus_2, psi_m_minus_1):
    """
    Calculate psi_m(x) using recursion relation.
    """
    return np.sqrt(2 / m) * x * psi_m_minus_1 - np.sqrt((m - 1) / m) * psi_m_minus_2


def _H_analytic(n, x):
    """
    Analytic expressions for psi_n(x) for 0 <= n <= 5.
    """

    if n == 0:
        return np.pi ** (-1 / 4) * np.exp(-(x ** 2) / 2)
    if n == 1:
        return np.sqrt(2) * np.pi ** (-1 / 4) * x * np.exp(-(x ** 2) / 2)
    if n == 2:
        return (
            (np.sqrt(2) * np.pi ** (1 / 4)) ** (-1)
            * (2 * x ** 2 - 1)
            * np.exp(-(x ** 2) / 2)
        )
    if n == 3:
        return (
            (np.sqrt(3) * np.pi ** (1 / 4)) ** (-1)
            * (2 * x ** 3 - 3 * x)
            * np.exp(-(x ** 2) / 2)
        )
    if n == 4:
        return (
            (2 * np.sqrt(6) * np.pi ** (1 / 4)) ** (-1)
            * (4 * x ** 4 - 12 * x ** 2 + 3)
            * np.exp(-(x ** 2) / 2)
        )
    if n == 5:
        return (
            (2 * np.sqrt(15) * np.pi ** (1 / 4)) ** (-1)
            * (4 * x ** 5 - 20 * x ** 3 + 15 * x)
            * np.exp(-(x ** 2) / 2)
        )
    raise ValueError("n must be an integer between 0 and 5")


def _H_const(n, all_n=False):
    """
    get the const in front of the Hermite function psi_n(x)
    Note that Hermite fn is given by psi_n(x) = const * H_n(x) * exp(-x^2/2)
    """
    if not all_n:
        return (
            1
            / np.sqrt(2 ** n * factorial(n))
            * np.pi ** (-1 / 4)
        )
    else:
        consts = []
        for i in range(n+1):
            const_curr_n = _H_const(i, all_n=False)
            consts.append(const_curr_n)
        return np.array(consts)


def _H_direct(n, x):
    """
    Calculate psi_n(x) using explicit definition.
    """
    const = _H_const(n, all_n=False)
    return const * np.exp(-(x ** 2) / 2) * eval_hermite(n, x)
    

def _H_coef5(n):
    """
    returns the coefficients up to the 5th order, of size (6,)
    """
    if n == 0:
        return np.array([1, 0, -0.5, 0, 1./8, -1./48]) * _H_const(n)
    if n == 1:
        return np.array([0, 2, 0, -1, 0, 1./4]) * _H_const(n)
    if n == 2:
        return np.array([-2, 0, 5, 0, -9./4, 0]) * _H_const(n)
    if n == 3:
        return np.array([-0, -12, 0, 14, 0, -5.5]) * _H_const(n)
    if n ==4:
        return np.array([12, 0, -54, 0, 41.5, 0]) * _H_const(n)
    if n == 5:
        return np.array([0, 120, 0, -220, 0, 127]) * _H_const(n)
    
    raise ValueError("n must be an integer between 0 and 5")


def hermite_function_linear_coefs(n, all_n=True):
    
    coef_linear_vec = hermite_polynomial_coefs(n, all_n, normalize=False)[1]
    consts = _H_const(n, all_n)
    return coef_linear_vec * consts


def hermite_functions_approx(n, x, all_n=True, order=5):

    assert n <= 5, f'n must be an integer between 0 and 5'
    assert order <= 5, f'order must be an integer between 0 and 5'
    
    bases = np.stack([x**i for i in range(order+1)],axis=0) ## (order+1, x.shape)
    if not all_n:
        poly_coefs = _H_coef5(n)[:(order+1)].reshape(1,-1) ## (1, order+1)
        fn_eval_approx = poly_coefs @ bases
        return fn_eval_approx
    else:
        fn_eval_approx_all_n = []
        for i in range(n+1):
            fn_eval_approx_all_n.append(hermite_functions_approx(i, x, False, order))
        return np.concatenate(fn_eval_approx_all_n,axis=0)
            
            
###########################
## the functions below are for Hermite polynomials
###########################

def hermite_polynomials(n, x, all_n=True, normalize=True):
    """
    calculate the mermite polynomials up to the nth order at position x, H_n(x).
    if all_n == True, then return all Hermite polynomials up to n
    if all_n == False, only return nth Hermite polynomial
    """
    if all_n:
        return _Hermite_poly_all_n(n, x, normalize)
    else:
        return _Hermite_poly_single_n(n, x, normalize)
    

def _Hermite_poly_all_n(n, x, normalize):
    
    try:
        H_ns = np.zeros((n + 1,) + x.shape)
    except AttributeError:  # x does not have property 'shape'
        H_ns = np.zeros((n + 1, 1))
    
    for order_id in range(n+1):
        H_ns[order_id] = _Hermite_poly_single_n(order_id, x, normalize)
    
    return H_ns


def _Hermite_poly_single_n(n, x, normalize=False):
    
    val = eval_hermite(n, x)
    if normalize:
        leading_coef = hermite(n, monic=False).c[0]
        val /= leading_coef
    return np.array(val)


def hermite_polynomial_coefs(n, all_n=True, normalize=True):
    """
    get the coefficients for H_n; see, e.g., https://francisbach.com/hermite-polynomials/
    Return: np.array
    - if all_n == False, size=(n+1, 1), where the ith element corresponds to the coef of x^i in H_n(x)
    - if all_n == True, size=(n+1, n+1)
        * coef_mtx[:, k] (i.e., the kth column) corresponds to the coef of H_k()
        * coef_mtx[j, :] (i.e., the jth row) corresponds to the coefficient of the j-th order term for each H_k(x), k=0,...,n
        * zero is padded towrad the end when k < n
        * the following holds by definition: sum(coef_mtx[:,k] * np.array([0, x, ..., x**n])) == eval_hermite(n,x) (before normalizing the leading coef to 1)
    """
    n_list = [n] if not all_n else list(range(n+1))
    
    coef_mtx = []
    for order_id in n_list:
        poly = hermite(order_id, monic=False)
        coef = poly.c[::-1]
        if normalize:
            coef = coef/coef[-1]
        
        if order_id < n:
            pad = np.zeros((n-order_id,))
            coef = np.concatenate([coef, pad],axis=0)
        coef_mtx.append(coef)
    
    coef_mtx = np.stack(coef_mtx, axis=1)
    return coef_mtx

def hermite_polynomial_linear_coefs(n, all_n=True, normalize=True):

    coef_linear_vec = hermite_polynomial_coefs(n, all_n, normalize=normalize)[1]
    return coef_linear_vec
