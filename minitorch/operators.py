"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul +
# - id +
# - add +
# - neg +
# - lt +
# - eq +
# - max +
# - is_close +
# - sigmoid +
# - relu +
# - log +
# - exp +
# - log_back +
# - inv +
# - inv_back +
# - relu_back +
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return float(x < y)


def eq(x: float, y: float) -> float:
    return float(x == y)


def max(x: float, y: float) -> float:
    return x if x > y else y


def min(x: float, y: float) -> float:
    return x if x < y else y


def is_close(x: float, y: float) -> bool:
    return (max(x, y) - min(x, y)) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + exp(neg(x)))
    return exp(x) / (1 + exp(x))


def relu(x: float) -> float:
    return float(max(x, 0))


def log(x: float) -> float:
    return math.log(x + 1e-6)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return mul(d, inv(x))


def inv(x: float) -> float:
    if x == 0:
        return 1 / (x + 1e-6)
    return 1 / x


def inv_back(x: float, d: float) -> float:
    return mul(d, neg(inv(mul(x, x))))


def relu_back(x: float, d: float) -> float:
    return mul(d, 0 if x <= 0 else 1)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float], arr: Iterable[float]) -> Iterable[float]:
    for val in arr:
        yield func(val)


def zipWith(func: Callable[[float, float], float], arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    a_iter = iter(arr1)
    b_iter = iter(arr2)

    val_a = 0.0
    val_b = 0.0

    alive_a = True
    alive_b = True
    try:
        val_a = next(a_iter)
    except StopIteration:
        alive_a = False
    try:
        val_b = next(b_iter)
    except StopIteration:
        alive_b = False

    while alive_a or alive_b:
        if alive_a and alive_b:
            yield func(val_a, val_b)
            try:
                val_a = next(a_iter)
            except StopIteration:
                alive_a = False
            try:
                val_b = next(b_iter)
            except StopIteration:
                alive_b = False
        elif alive_a and not alive_b:
            yield func(val_a, 0)
            try:
                val_a = next(a_iter)
            except StopIteration:
                alive_a = False
        else:
            yield func(0, val_b)
            try:
                val_b = next(b_iter)
            except StopIteration:
                alive_b = False


def reduce(arr1: Iterable[float], func: Callable[[float, float], float], start: float) -> float:
    for val in arr1:
        start = func(start, val)
    return start


def negList(arr: Iterable[float]) -> Iterable[float]:
    return list(map(neg, arr))


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    return list(zipWith(add, arr1, arr2))


def sum(arr: Iterable[float]) -> float:
    return reduce(arr, add, 0)


def prod(arr: Iterable[float]) -> float:
    return reduce(arr, mul, 1)
