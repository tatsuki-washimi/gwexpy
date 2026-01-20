from inspect import Parameter, signature
from typing import Any

import numpy as np


class Model:
    """Base class for models to support unit propagation and iminuit interoperability."""

    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError


class Polynomial(Model):
    """
    Polynomial model: f(x) = c0 + c1*x + ... + cn*x^n
    """

    def __init__(self, degree: int):
        self.degree = degree
        self.param_names = [f"p{i}" for i in range(degree + 1)]

        # Construct signature for iminuit
        params = [Parameter("x", Parameter.POSITIONAL_OR_KEYWORD)]
        for name in self.param_names:
            params.append(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD))
        self.__signature__ = signature(lambda: None).replace(parameters=params)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        # Handle both positional and keyword arguments from iminuit
        if kwargs:
            p_vals = [kwargs[name] for name in self.param_names]
        else:
            p_vals = list(args)

        res = 0
        for i, p in enumerate(p_vals):
            res += p * x**i
        return res



def gaussian(x, A, mu, sigma):
    """
    Gaussian distribution.
    f(x) = A * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def exponential(x, A, tau):
    """
    Exponential decay.
    f(x) = A * exp(-x / tau)
    """
    return A * np.exp(-x / tau)


def power_law(x, A, alpha):
    """
    Power law.
    f(x) = A * x^alpha
    """
    return A * x**alpha


def damped_oscillation(x, A, tau, f, phi=0):
    """
    Damped oscillation.
    f(x) = A * exp(-x / tau) * sin(2 * pi * f * x + phi)
    """
    return A * np.exp(-x / tau) * np.sin(2 * np.pi * f * x + phi)


def landau(x, A, mu, sigma):
    """
    Landau distribution approximation (Moyal distribution).
    f(x) = A * exp(-0.5 * (lambda + exp(-lambda))), where lambda = (x - mu) / sigma
    """
    lam = (x - mu) / sigma
    return A * np.exp(-0.5 * (lam + np.exp(-lam)))


def make_pol_func(n):
    """
    Create a polynomial function of degree n with signature (x, p0, p1, ..., pn).
    f(x) = p0 + p1*x + ... + pn*x^n
    """
    param_names = [f"p{i}" for i in range(n + 1)]

    def pol_func(x, *args, **kwargs):
        # Handle both positional and keyword arguments
        if kwargs:
            p_vals = [kwargs[name] for name in param_names]
        else:
            p_vals = list(args)

        res = 0
        for i, p in enumerate(p_vals):
            res += p * x**i
        return res

    # Construct the function signature manually so iminuit can detect parameter names
    parameters = [Parameter("x", Parameter.POSITIONAL_OR_KEYWORD)]
    for name in param_names:
        parameters.append(Parameter(name, Parameter.POSITIONAL_OR_KEYWORD))

    pol_func.__signature__ = signature(lambda: None).replace(parameters=parameters)  # type: ignore[attr-defined]
    pol_func.__doc__ = f"Polynomial of degree {n}: p0 + p1*x + ... + p{n}*x^{n}"
    return pol_func


# Predefined dictionary
MODELS: dict[str, Any] = {
    "gaus": gaussian,
    "gaussian": gaussian,
    "exp": exponential,
    "expo": exponential,  # ROOT style
    "exponential": exponential,
    "landau": landau,
    "power_law": power_law,
    "damped_oscillation": damped_oscillation,
}

# Add pol0 to pol9 as Polynomial instances
for i in range(10):
    MODELS[f"pol{i}"] = Polynomial(i)


def get_model(name: str | Any) -> Any:
    """
    Retrieve a model function or object by name (case-insensitive).
    """
    if not isinstance(name, str):
        if callable(name):
            return name
        return None

    key = name.lower()
    if key in MODELS:
        return MODELS[key]

    # Handle polN for N >= 10
    if key.startswith("pol") and key[3:].isdigit():
        n = int(key[3:])
        return Polynomial(n)

    raise ValueError(f"Unknown model name: {name}. Available: {list(MODELS.keys())}")
