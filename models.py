from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Union

from pydantic import BaseModel


class RunKind(StrEnum):
    UNIT = "unit"
    INTEGRATION = "integration"
    MUTATION = "mutation"


class RunMeta(BaseModel):
    id: str
    projectId: str
    commitSha: str
    regionId: str | None = None
    timestamp: datetime


class TestRunResult(RunMeta):
    kind: Literal[RunKind.UNIT, RunKind.INTEGRATION]
    passed: bool
    totalTests: int
    passedTests: int
    failedTests: int
    durationMs: int
    framework: str
    failingTests: list[str]


class FileMutationStats(BaseModel):
    filePath: str
    totalMutants: int
    killed: int
    survived: int
    noCoverage: int
    mutationScore: float


class MutationRunResult(RunMeta):
    kind: Literal[RunKind.MUTATION]
    tool: str
    totalMutants: int
    killed: int
    survived: int
    noCoverage: int
    mutationScore: float
    runtimeMs: int
    byFile: list[FileMutationStats]


class PIDComponents(BaseModel):
    p: float
    i: float
    d: float


class RegionMetrics(BaseModel):
    projectId: str
    commitSha: str
    regionId: str
    mutationScore: float
    centrality: float
    triviality: float
    pid: PIDComponents


RunResult = Union[TestRunResult, MutationRunResult]


class RunHistory(BaseModel):
    projectId: str
    regionId: str | None = None
    runs: list[RunResult]
