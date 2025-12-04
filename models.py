from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class RunKind(str, Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    MUTATION = "mutation"


class RunMeta(BaseModel):
    id: str
    projectId: str
    commitSha: str
    regionId: Optional[str] = None
    timestamp: datetime


class TestRunResult(RunMeta):
    kind: Literal[RunKind.UNIT, RunKind.INTEGRATION]
    passed: bool
    totalTests: int
    passedTests: int
    failedTests: int
    durationMs: int
    framework: str
    failingTests: List[str]


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
    byFile: List[FileMutationStats]


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
    regionId: Optional[str] = None
    runs: List[RunResult]
