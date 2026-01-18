"""Finite field arithmetic helpers for zk-friendly computations."""

from __future__ import annotations

FieldInt = int

# FIELD_MODULUS: FieldInt = 2**255 - 19  # Ed25519 prime, fits common zk backends.
FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617



class FieldElement:
    """Field element with automatic modulo arithmetic."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value % FIELD_MODULUS

    def __add__(self, other: "FieldElement | int") -> "FieldElement":
        return FieldElement(self.value + _coerce(other))

    def __sub__(self, other: "FieldElement | int") -> "FieldElement":
        return FieldElement(self.value - _coerce(other))

    def __mul__(self, other: "FieldElement | int") -> "FieldElement":
        return FieldElement(self.value * _coerce(other))

    def __truediv__(self, other: "FieldElement | int") -> "FieldElement":
        return self * FieldElement(_coerce(other)).inverse()

    def __pow__(self, exponent: int) -> "FieldElement":
        return FieldElement(pow(self.value, exponent, FIELD_MODULUS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (FieldElement, int)):
            return False
        return self.value == _coerce(other)

    def __repr__(self) -> str:
        return f"FieldElement({self.value})"

    def inverse(self) -> "FieldElement":
        if self.value == 0:
            raise ZeroDivisionError("cannot invert 0 in field")
        return FieldElement(pow(self.value, FIELD_MODULUS - 2, FIELD_MODULUS))

    def sqrt(self) -> "FieldElement":
        # Tonelli-Shanks simplified for p % 4 == 1 (true for this modulus).
        if self.value == 0:
            return FieldElement(0)
        candidate = pow(self.value, (FIELD_MODULUS + 3) // 4, FIELD_MODULUS)
        if (candidate * candidate).value != self.value:
            raise ValueError("no square root exists for value in this field")
        return FieldElement(candidate)


def _coerce(value: "FieldElement | int") -> int:
    return value.value if isinstance(value, FieldElement) else value
