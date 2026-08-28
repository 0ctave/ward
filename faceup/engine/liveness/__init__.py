"""Liveness voters (DESIGN.md §7). Each returns a Vote; ABSTAIN never counts against the user."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Result(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"


@dataclass
class Vote:
    voter: str
    result: Result
    reason: str = ""
    metric: float | None = None

    def __str__(self) -> str:
        return f"{self.voter}:{self.result.value}"


def PASS(voter, reason="", metric=None):    return Vote(voter, Result.PASS, reason, metric)
def FAIL(voter, reason="", metric=None):    return Vote(voter, Result.FAIL, reason, metric)
def ABSTAIN(voter, reason="", metric=None): return Vote(voter, Result.ABSTAIN, reason, metric)
