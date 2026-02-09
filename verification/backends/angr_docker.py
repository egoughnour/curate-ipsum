"""
angr Docker-based symbolic execution backend.

Runs angr inside a Docker container, communicating via JSON files
mounted into the container.

Build the image with:  docker compose --profile verify build
Run standalone:        docker compose run --rm angr-runner

The runner script (verification/runners/run_angr.py) is baked into
the image via docker/Dockerfile.angr-runner.

Decision: D-016
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from verification.backend import VerificationBackend
from verification.types import (
    VerificationRequest,
    VerificationResult,
    VerificationStatus,
)

LOG = logging.getLogger("verification.backends.angr_docker")

# Default: the compose-built image (docker compose --profile verify build)
# Falls back to upstream angr/angr if the compose image isn't built yet.
DEFAULT_ANGR_IMAGE = os.environ.get("ANGR_DOCKER_IMAGE", "curate-ipsum-angr-runner")
# Path to the runner script inside the container (baked in by Dockerfile)
DEFAULT_RUNNER_SCRIPT = "/opt/runner/run_angr.py"


class AngrDockerBackend(VerificationBackend):
    """
    Verification via angr symbolic execution in Docker.

    Workflow:
    1. Write VerificationRequest JSON to a temp directory
    2. Run Docker container with request + binary mounted
    3. Read VerificationResult JSON from output
    4. Clean up temp artifacts

    Requires Docker to be available on the host.
    """

    def __init__(
        self,
        docker_image: str = DEFAULT_ANGR_IMAGE,
        runner_script: str | None = None,
        runner_script_host_path: str | None = None,
        work_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._image = docker_image
        self._runner_script = runner_script or DEFAULT_RUNNER_SCRIPT
        self._runner_host_path = runner_script_host_path
        self._work_dir = work_dir or tempfile.gettempdir()

    def supports(self) -> dict[str, Any]:
        return {
            "input": "binary",
            "constraints": ["comparison"],
            "find": ["addr_reached"],
            "avoid": ["addr_avoided"],
        }

    async def verify(self, request: VerificationRequest) -> VerificationResult:
        t0 = time.monotonic()

        # Create deterministic artifact directory based on request hash
        req_hash = hashlib.sha256(request.to_json().encode()).hexdigest()[:12]
        artifact_dir = Path(self._work_dir) / f"angr_run_{req_hash}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        req_path = artifact_dir / "request.json"
        resp_path = artifact_dir / "response.json"

        try:
            # Write request JSON
            req_path.write_text(request.to_json(), encoding="utf-8")

            # Build Docker command
            cmd = self._build_docker_cmd(request, artifact_dir, req_path, resp_path)
            LOG.info("angr Docker: running %s", " ".join(cmd[:6]) + " ...")

            # Execute with timeout (budget + 30s grace period)
            timeout = request.budget.timeout_s + 30
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return VerificationResult(
                    status=VerificationStatus.NO_CE_WITHIN_BUDGET,
                    stats={"elapsed_s": time.monotonic() - t0, "timeout": True},
                    logs="Docker process killed after timeout",
                )

            # Read response
            if not resp_path.exists():
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    stats={"elapsed_s": time.monotonic() - t0, "exit_code": proc.returncode},
                    logs=f"No response file. stderr: {stderr.decode(errors='replace')[:500]}",
                )

            resp_data = json.loads(resp_path.read_text(encoding="utf-8"))
            result = VerificationResult.from_dict(resp_data)
            result.stats["elapsed_s"] = time.monotonic() - t0
            return result

        except FileNotFoundError:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0},
                logs="Docker not found. Install Docker to use angr backend.",
            )
        except Exception as exc:
            LOG.exception("angr Docker backend error")
            return VerificationResult(
                status=VerificationStatus.ERROR,
                stats={"elapsed_s": time.monotonic() - t0},
                logs=f"angr Docker error: {exc}",
            )
        finally:
            # Clean up artifacts (best-effort)
            try:
                shutil.rmtree(artifact_dir, ignore_errors=True)
            except Exception:
                pass

    def _build_docker_cmd(
        self,
        request: VerificationRequest,
        artifact_dir: Path,
        req_path: Path,
        resp_path: Path,
    ) -> list:
        """Build the Docker run command."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "2g",
            "--cpus",
            "2",
            # Mount artifact directory
            "-v",
            f"{artifact_dir}:/work",
        ]

        # Mount the runner script if provided on host
        if self._runner_host_path:
            cmd.extend(["-v", f"{self._runner_host_path}:/runner/run_angr.py:ro"])

        # Mount binary directory — the binary_name in the request
        # should resolve to /bin_in/<binary_name> inside container
        binary_dir = str(Path(request.target_binary).parent)
        _binary_name = Path(request.target_binary).name
        if os.path.isfile(request.target_binary):
            cmd.extend(["-v", f"{binary_dir}:/bin_in:ro"])
        else:
            # Binary path is just a name; assume it's already in the image
            # or will be provided by a pre-built runner image
            pass

        # The compose-built image has ENTRYPOINT [run_angr.py] so just pass paths.
        # For the upstream angr/angr image, invoke the runner script explicitly.
        cmd.append(self._image)
        if self._image == "curate-ipsum-angr-runner":
            # Compose-built: ENTRYPOINT already set, just pass args
            cmd.extend(["/work/request.json", "/work/response.json"])
        else:
            # Upstream image: run the runner script explicitly
            cmd.extend(
                [
                    "python3",
                    self._runner_script,
                    "/work/request.json",
                    "/work/response.json",
                ]
            )

        return cmd
