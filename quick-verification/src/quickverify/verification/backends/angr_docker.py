"""AngrBackendDocker — runs angr in a container, exchanges JSON via mounted volumes.

This is the "medium" tier in the CEGAR chain (Z3 → angr → KLEE).
Ported from angr_adapter_baseline/verification/backends/angr_docker.py
with improvements for the main package.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import shutil
import subprocess
from typing import Optional

from ..backend import IVerificationBackend
from ..types import VerificationRequest, VerificationResult

log = logging.getLogger(__name__)


class AngrBackendDocker(IVerificationBackend):
    """Docker-first angr backend.

    Runs the angr runner script inside the ``angr/angr`` Docker image.
    Communication is via JSON files mounted into the container.

    Args:
        image: Docker image to use (default: ``angr/angr``).
        artifacts_dir: Directory for per-run artifacts (request/response/logs).
        runner_script: Path to the ``run_angr.py`` runner script.
                       Copied into each run directory so the container can execute it.
        docker_bin: Path to the ``docker`` binary.
        extra_docker_args: Additional ``docker run`` arguments (e.g., resource limits).
    """

    def __init__(
        self,
        image: str = "angr/angr",
        artifacts_dir: str = "artifacts/verify",
        runner_script: Optional[str] = None,
        docker_bin: str = "docker",
        extra_docker_args: Optional[list] = None,
    ):
        self.image = image
        self.artifacts_dir = pathlib.Path(artifacts_dir)
        self.docker_bin = docker_bin
        self.extra_docker_args = extra_docker_args or []

        # Locate runner script: explicit, or relative to package
        if runner_script:
            self.runner_script = pathlib.Path(runner_script)
        else:
            # Default: look for the baseline runner
            pkg_root = pathlib.Path(__file__).resolve().parents[3]
            candidates = [
                pkg_root / "verification" / "runners" / "run_angr.py",
                pathlib.Path("angr_adapter_baseline") / "verification" / "runners" / "run_angr.py",
            ]
            self.runner_script = next((c for c in candidates if c.exists()), candidates[0])

    def supports(self) -> dict:
        return {
            "input": "binary",
            "constraints": ["mini-dsl"],
            "find": ["addr_reached"],
            "avoid": ["addr_avoided"],
        }

    def verify(self, req: VerificationRequest) -> VerificationResult:
        run_dir = self._prepare_run_dir(req)

        # Copy runner into run dir
        runner_dest = run_dir / "run_angr.py"
        if self.runner_script.exists():
            shutil.copy2(self.runner_script, runner_dest)
        else:
            return VerificationResult(
                status="error",
                stats={},
                logs=f"Runner script not found: {self.runner_script}",
            )

        # Write request JSON
        req_path = run_dir / "request.json"
        resp_path = run_dir / "response.json"
        req_path.write_text(json.dumps(req.to_json(), indent=2), encoding="utf-8")

        # Resolve binary directory for mounting
        bin_dir = pathlib.Path(req.target.binary_name).parent
        if not bin_dir.is_absolute():
            bin_dir = pathlib.Path(".").resolve()

        cmd = [
            self.docker_bin, "run", "--rm",
            *self.extra_docker_args,
            "-v", f"{run_dir.resolve()}:/work",
            "-v", f"{bin_dir.resolve()}:/bin_in:ro",
            self.image,
            "python", "/work/run_angr.py", "/work/request.json", "/work/response.json",
        ]

        log.info("Running: %s", " ".join(cmd))
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=req.budget.timeout_s + 30,  # grace period
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                status="no_ce_within_budget",
                stats={"docker_timeout": True},
                logs="Docker process timed out",
            )

        # Save logs
        (run_dir / "stdout.log").write_text(p.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(p.stderr, encoding="utf-8")

        if not resp_path.exists():
            return VerificationResult(
                status="error",
                stats={"returncode": p.returncode},
                logs=(p.stderr[:2000] if p.stderr else "No response.json produced"),
            )

        # Parse response
        try:
            data = json.loads(resp_path.read_text(encoding="utf-8"))
            return VerificationResult.from_json(data)
        except (json.JSONDecodeError, KeyError) as e:
            return VerificationResult(
                status="error",
                stats={"returncode": p.returncode},
                logs=f"Response parse error: {e}",
            )

    def _prepare_run_dir(self, req: VerificationRequest) -> pathlib.Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(
            json.dumps(req.to_json(), sort_keys=True).encode()
        ).hexdigest()[:16]
        d = self.artifacts_dir / h
        d.mkdir(parents=True, exist_ok=True)
        return d
