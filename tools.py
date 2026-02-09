from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from models import (
    FileMutationStats,
    MutationRunResult,
    PIDComponents,
    RegionMetrics,
    RunHistory,
    RunKind,
    RunResult,
    TestRunResult,
)

LOG = logging.getLogger("mutation_tool")


DATA_DIR = Path(os.environ.get("MUTATION_TOOL_DATA_DIR", ".mutation_tool_data"))
RUNS_FILE = DATA_DIR / "runs.jsonl"
DEFAULT_STRYKER_REPORT = os.environ.get("MUTATION_TOOL_STRYKER_REPORT", "reports/mutation/mutation.json")

PID_WINDOW = int(os.environ.get("MUTATION_TOOL_PID_WINDOW", "5"))
PID_DECAY = float(os.environ.get("MUTATION_TOOL_PID_DECAY", "0.8"))


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


async def run_command(command: str, working_directory: str, timeout: float | None = None) -> CommandResult:
    """Run a shell command and capture stdout, stderr, exit code, and duration."""
    cwd_path = Path(working_directory)
    if not cwd_path.exists() or not cwd_path.is_dir():
        raise ValueError(f"Working directory does not exist: {working_directory}")

    start = time.perf_counter()
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd_path),
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        duration_ms = int((time.perf_counter() - start) * 1000)
        return CommandResult(
            exit_code=-1,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace") + "\nProcess timed out",
            duration_ms=duration_ms,
            timed_out=True,
        )

    duration_ms = int((time.perf_counter() - start) * 1000)
    return CommandResult(
        exit_code=process.returncode,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        duration_ms=duration_ms,
    )


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Flag to enable/disable BRS dual-write
BRS_DUAL_WRITE = os.environ.get("CURATE_IPSUM_BRS_DUAL_WRITE", "1") == "1"


def _dual_write_to_brs(run: RunResult) -> None:
    """
    Write run result to BRS CASStore as evidence.

    This enables belief revision operations on test/mutation results.
    Failures are logged but do not block the primary JSONL write.
    """
    if not BRS_DUAL_WRITE:
        return

    try:
        from adapters.evidence_adapter import (
            mutation_result_to_evidence,
            test_result_to_evidence,
        )
        from theory import TheoryManager
    except ImportError:
        LOG.debug("BRS dual-write skipped: py-brs not installed")
        return

    try:
        manager = TheoryManager(DATA_DIR / run.projectId)

        if isinstance(run, MutationRunResult):
            evidence = mutation_result_to_evidence(run)
        else:
            evidence = test_result_to_evidence(run)

        manager.store_evidence(evidence)
        LOG.debug("Stored evidence %s in BRS", evidence.id)

    except Exception as exc:
        # Don't fail the primary operation - just log
        LOG.warning("BRS dual-write failed: %s", exc)


def append_run(run: RunResult) -> None:
    """
    Append a run result to storage.

    Primary: JSONL file (runs.jsonl)
    Secondary: BRS CASStore (if enabled and py-brs installed)
    """
    _ensure_data_dir()

    # Primary: JSONL write
    payload = run.model_dump(mode="json")
    with RUNS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    # Secondary: BRS dual-write
    _dual_write_to_brs(run)


def _deserialize_run(record: dict) -> RunResult | None:
    kind = record.get("kind")
    try:
        if kind == RunKind.MUTATION:
            return MutationRunResult.model_validate(record)
        if kind in (RunKind.UNIT, RunKind.INTEGRATION):
            return TestRunResult.model_validate(record)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to parse run record: %s", exc, exc_info=True)
        return None
    return None


def _load_runs() -> list[RunResult]:
    if not RUNS_FILE.exists():
        return []
    runs: list[RunResult] = []
    with RUNS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("Skipping invalid JSON line in history")
                continue
            parsed = _deserialize_run(record)
            if parsed:
                runs.append(parsed)
    return runs


def get_run_history(project_id: str, region_id: str | None, limit: int | None) -> RunHistory:
    runs = _load_runs()
    filtered = [
        run
        for run in runs
        if run.projectId == project_id and (region_id is None or getattr(run, "regionId", None) == region_id)
    ]
    filtered.sort(key=lambda r: r.timestamp, reverse=True)
    limited = filtered if limit is None else filtered[:limit]
    return RunHistory(projectId=project_id, regionId=region_id, runs=limited)


TEST_TOTAL_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"Total tests:\s*(\d+).+Passed:\s*(\d+).+Failed:\s*(\d+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Tests run:\s*(\d+)\s*,\s*Passed:\s*(\d+)\s*,\s*Failed:\s*(\d+)", re.IGNORECASE),
)

FAILING_NAME_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"Failed\s+(?P<name>[\w\.\:\/\-]+)", re.IGNORECASE),
    re.compile(r"\[FAIL\]\s+(?P<name>[\w\.\:\/\-]+)", re.IGNORECASE),
)


def parse_test_output(stdout: str, stderr: str) -> tuple[int, int, int, list[str]]:
    combined = f"{stdout}\n{stderr}"
    total = passed = failed = 0
    for pattern in TEST_TOTAL_PATTERNS:
        match = pattern.search(combined)
        if match:
            total, passed, failed = (int(match.group(i)) for i in range(1, 4))
            break

    failing_tests: list[str] = []
    for line in combined.splitlines():
        for pattern in FAILING_NAME_PATTERNS:
            match = pattern.search(line)
            if match:
                failing_tests.append(match.group("name"))

    return total, passed, failed, failing_tests


def _compute_mutation_score(killed: int, survived: int, no_coverage: int) -> float:
    denominator = killed + survived
    if denominator == 0:
        return 0.0
    return killed / float(denominator)


def _collect_mutants(data: dict) -> dict[str, list[dict]]:
    files: dict[str, list[dict]] = {}
    if "files" in data and isinstance(data["files"], dict):
        for path, file_info in data["files"].items():
            mutants = file_info.get("mutants", [])
            if isinstance(mutants, list):
                files[path] = mutants
    elif "mutantResults" in data and isinstance(data["mutantResults"], list):
        files["<all>"] = data["mutantResults"]
    return files


def _count_mutants(mutants: Iterable[dict]) -> tuple[int, int, int]:
    killed = survived = no_coverage = 0
    for mutant in mutants:
        status = str(mutant.get("status", "")).lower()
        if status == "killed":
            killed += 1
        elif status == "survived":
            survived += 1
        elif status in ("no coverage", "nocoverage", "survivednocoverage"):
            no_coverage += 1
    return killed, survived, no_coverage


def parse_stryker_output(
    report_path: str | None, working_directory: str
) -> tuple[int, int, int, int, float, list[FileMutationStats]]:
    candidate_paths = []
    cwd_path = Path(working_directory)
    if report_path:
        explicit_path = Path(report_path)
        candidate_paths.append(explicit_path)
        if not explicit_path.is_absolute():
            candidate_paths.append(cwd_path / explicit_path)
    if DEFAULT_STRYKER_REPORT:
        candidate_paths.append(cwd_path / DEFAULT_STRYKER_REPORT)
    candidate_paths.append(cwd_path / "reports" / "mutation" / "mutation.json")
    candidate_paths.append(cwd_path / "reports" / "stryker-report.json")

    report_file = next((p for p in candidate_paths if p.exists()), None)
    if not report_file:
        raise FileNotFoundError("Stryker report not found in expected locations")

    with report_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Invalid Stryker report format")

    files = _collect_mutants(data)
    by_file: list[FileMutationStats] = []
    total_killed = total_survived = total_no_coverage = total_mutants = 0

    for file_path, mutants in files.items():
        killed, survived, no_cov = _count_mutants(mutants)
        total = len(mutants)
        total_mutants += total
        total_killed += killed
        total_survived += survived
        total_no_coverage += no_cov
        score = _compute_mutation_score(killed, survived, no_cov)
        by_file.append(
            FileMutationStats(
                filePath=str(file_path),
                totalMutants=total,
                killed=killed,
                survived=survived,
                noCoverage=no_cov,
                mutationScore=score,
            )
        )

    score_from_report = data.get("mutationScore")
    if isinstance(score_from_report, (int, float)):
        reported_score = float(score_from_report)
        if reported_score > 1:
            reported_score /= 100.0
        mutation_score = reported_score
    else:
        mutation_score = _compute_mutation_score(total_killed, total_survived, total_no_coverage)

    return total_mutants, total_killed, total_survived, total_no_coverage, mutation_score, by_file


def compute_region_metrics(project_id: str, commit_sha: str, region_id: str, history: list[RunResult]) -> RegionMetrics:
    mutation_runs = [run for run in history if isinstance(run, MutationRunResult) and run.regionId == region_id]
    if not mutation_runs:
        raise ValueError("No mutation runs available for region metrics")

    mutation_runs.sort(key=lambda r: r.timestamp)
    scores = [run.mutationScore for run in mutation_runs]

    p_term = 1.0 - scores[-1]

    inverted = [1.0 - score for score in scores[-PID_WINDOW:]]
    i_term = 0.0
    for idx, value in enumerate(reversed(inverted)):
        i_term += value * (PID_DECAY**idx)

    d_term = 0.0
    if len(inverted) >= 2:
        d_term = inverted[-1] - inverted[-2]

    pid = PIDComponents(p=p_term, i=i_term, d=d_term)

    return RegionMetrics(
        projectId=project_id,
        commitSha=commit_sha,
        regionId=region_id,
        mutationScore=scores[-1],
        centrality=0.5,
        triviality=0.5,
        pid=pid,
    )


async def _execute_test_run(
    kind: RunKind,
    project_id: str,
    commit_sha: str,
    command: str,
    working_directory: str,
    region_id: str | None,
    framework: str,
) -> TestRunResult:
    if kind not in (RunKind.UNIT, RunKind.INTEGRATION):
        raise ValueError("Invalid test run kind")

    result = await run_command(command, working_directory)
    total_tests, passed_tests, failed_tests, failing_tests = parse_test_output(result.stdout, result.stderr)

    if total_tests == 0:
        total_tests = passed_tests + failed_tests
    passed_flag = result.exit_code == 0 and failed_tests == 0 and not result.timed_out

    test_run = TestRunResult(
        id=str(uuid4()),
        projectId=project_id,
        commitSha=commit_sha,
        regionId=region_id,
        timestamp=datetime.now(UTC),
        kind=kind,
        passed=passed_flag,
        totalTests=total_tests,
        passedTests=passed_tests if passed_tests else (total_tests - failed_tests),
        failedTests=failed_tests,
        durationMs=result.duration_ms,
        framework=framework,
        failingTests=failing_tests,
    )

    append_run(test_run)
    return test_run


async def run_unit_tests(
    projectId: str,
    commitSha: str,
    command: str,
    workingDirectory: str,
    regionId: str | None = None,
    framework: str = "generic",
) -> TestRunResult:
    return await _execute_test_run(
        kind=RunKind.UNIT,
        project_id=projectId,
        commit_sha=commitSha,
        command=command,
        working_directory=workingDirectory,
        region_id=regionId,
        framework=framework,
    )


async def run_integration_tests(
    projectId: str,
    commitSha: str,
    command: str,
    workingDirectory: str,
    regionId: str | None = None,
    framework: str = "generic",
) -> TestRunResult:
    return await _execute_test_run(
        kind=RunKind.INTEGRATION,
        project_id=projectId,
        commit_sha=commitSha,
        command=command,
        working_directory=workingDirectory,
        region_id=regionId,
        framework=framework,
    )


async def run_mutation_tests(
    projectId: str,
    commitSha: str,
    command: str,
    workingDirectory: str,
    regionId: str | None = None,
    tool: str | None = None,  # Now optional - auto-detected if None
    reportPath: str | None = None,
) -> MutationRunResult:
    """
    Run mutation tests and parse results.

    If tool is not specified, auto-detects based on project structure
    and available mutation output.
    """
    result = await run_command(command, workingDirectory)

    # Use unified parser with auto-detection
    from parsers import get_detected_tool, parse_mutation_output

    try:
        total_mutants, killed, survived, no_coverage, mutation_score, by_file = parse_mutation_output(
            working_directory=workingDirectory,
            tool=tool,
            report_path=reportPath,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to parse mutation output: %s", exc, exc_info=True)
        raise

    # Get the actual tool used (important when auto-detected)
    actual_tool = tool or get_detected_tool(workingDirectory) or "unknown"

    mutation_run = MutationRunResult(
        id=str(uuid4()),
        projectId=projectId,
        commitSha=commitSha,
        regionId=regionId,
        timestamp=datetime.now(UTC),
        kind=RunKind.MUTATION,
        tool=actual_tool,
        totalMutants=total_mutants,
        killed=killed,
        survived=survived,
        noCoverage=no_coverage,
        mutationScore=mutation_score,
        runtimeMs=result.duration_ms,
        byFile=by_file,
    )

    append_run(mutation_run)
    return mutation_run


def history_tool(projectId: str, regionId: str | None = None, limit: int | None = None) -> RunHistory:
    parsed_limit: int | None
    if limit is None:
        parsed_limit = None
    else:
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        if parsed_limit <= 0:
            parsed_limit = None
    return get_run_history(projectId, regionId, parsed_limit)


def region_metrics_tool(projectId: str, commitSha: str, regionId: str) -> RegionMetrics:
    history = get_run_history(projectId, regionId, None).runs
    return compute_region_metrics(projectId, commitSha, regionId, history)
