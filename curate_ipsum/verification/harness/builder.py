"""
Harness builder: generates minimal C harness binaries from function specifications.

Compiles with ``gcc -O0 -g -fno-inline`` to preserve symbol information
for angr symbolic execution.

Decision: D-016
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("verification.harness.builder")


@dataclass
class HarnessSpec:
    """Specification for generating a C harness."""

    function_name: str
    parameters: list[dict[str, str]] = field(default_factory=list)
    # Each param: {"name": "x", "type": "int", "bits": "32"}
    violation_condition: str = ""  # C expression for the violation
    includes: list[str] = field(default_factory=list)
    extra_code: str = ""


_HARNESS_TEMPLATE = """\
/* Auto-generated harness for verification */
{includes}

void violation(void) {{
    /* Marker function — angr searches for this address */
    return;
}}

{extra_code}

void {function_name}({params}) {{
    if ({violation_condition}) {{
        violation();
    }}
}}

int main(void) {{
    {function_name}({call_args});
    return 0;
}}
"""


class HarnessBuilder:
    """
    Builds C harness binaries from specifications.

    Usage::

        builder = HarnessBuilder()
        binary_path = builder.build(spec, output_dir="/tmp/harnesses")
    """

    def __init__(self, cc: str = "gcc", cflags: list[str] | None = None) -> None:
        self._cc = cc
        self._cflags = cflags or ["-O0", "-g", "-fno-inline", "-fno-stack-protector"]

    def generate_source(self, spec: HarnessSpec) -> str:
        """Generate C source code for the harness."""
        includes = "\n".join(f"#include <{h}>" for h in (spec.includes or ["stdio.h"]))

        params = ", ".join(f"{p.get('type', 'int')} {p['name']}" for p in spec.parameters) or "void"

        call_args = ", ".join("0" for _ in spec.parameters)

        violation = spec.violation_condition or "0"

        return _HARNESS_TEMPLATE.format(
            includes=includes,
            extra_code=spec.extra_code or "",
            function_name=spec.function_name,
            params=params,
            call_args=call_args,
            violation_condition=violation,
        )

    def build(
        self,
        spec: HarnessSpec,
        output_dir: str | None = None,
    ) -> str:
        """
        Generate and compile a harness binary.

        Returns the path to the compiled binary.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="harness_")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        src_path = out_path / f"{spec.function_name}_harness.c"
        bin_path = out_path / f"{spec.function_name}_harness"

        source = self.generate_source(spec)
        src_path.write_text(source, encoding="utf-8")

        cmd = [self._cc] + self._cflags + [str(src_path), "-o", str(bin_path)]
        LOG.info("Compiling harness: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Harness compilation failed (exit {result.returncode}):\n{result.stderr}")

        return str(bin_path)

    def build_from_spec(
        self,
        function_name: str,
        params: list[dict[str, str]],
        violation: str,
        output_dir: str | None = None,
    ) -> str:
        """Convenience: build a harness from individual arguments."""
        spec = HarnessSpec(
            function_name=function_name,
            parameters=params,
            violation_condition=violation,
        )
        return self.build(spec, output_dir)
