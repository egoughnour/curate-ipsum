"""
HarnessBuilder — transforms graph partitions into compiled harness binaries.

Generates a minimal C harness wrapping a target function, compiles it,
and returns a VerificationTarget for the angr adapter.
"""
from __future__ import annotations
import os
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class HarnessTarget:
    """Output of a successful harness build."""
    binary_name: str
    binary_path: str
    entry: str
    symbols: List[Dict[str, Any]]


# Template for a basic integer-args harness
HARNESS_TEMPLATE = """\
#include <stdint.h>
#include <stdlib.h>

__attribute__((noinline)) void violation(void) {{
    volatile int *p = (int*)0;
    *p = 1;
}}

__attribute__((noinline)) void ok_exit(void) {{ return; }}

{function_prototype}

__attribute__((noinline)) int target_wrapper({param_declarations}) {{
    {return_type} result = {function_call};
    {property_check}
    return 0;
}}

int main(int argc, char **argv) {{
    {declare_defaults}
    (void)target_wrapper({pass_args});
    ok_exit();
    return 0;
}}
"""


class HarnessBuilder:
    """Builds compiled C harness binaries from graph partition info."""

    def __init__(
        self,
        cc: str = "gcc",
        cflags: Optional[List[str]] = None,
        output_dir: str = "artifacts/harnesses",
    ):
        self.cc = cc
        self.cflags = cflags or ["-O0", "-g", "-fno-omit-frame-pointer", "-fno-inline"]
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        function_name: str,
        param_specs: List[Dict[str, Any]],
        property_check: str = "",
        return_type: str = "int",
        extra_source: str = "",
        harness_name: Optional[str] = None,
    ) -> HarnessTarget:
        """Generate, compile, and return a harness target.

        Args:
            function_name: Name of the function to wrap.
            param_specs: List of {"name": str, "c_type": str, "bits": int, "kind": str}.
            property_check: C code for the property check (use violation() to signal failure).
            return_type: Return type of the target function.
            extra_source: Additional C source to include (e.g., the target function itself).
            harness_name: Output binary name (default: function_name + "_harness").
        """
        name = harness_name or f"{function_name}_harness"

        # Generate parameter declarations
        param_decls = ", ".join(f"{p['c_type']} {p['name']}" for p in param_specs)
        pass_args = ", ".join(p["name"] for p in param_specs)
        declare_defaults = "\n    ".join(f"{p['c_type']} {p['name']} = 0;" for p in param_specs)
        function_call = f"{function_name}({pass_args})"

        # Build prototype
        proto_params = ", ".join(p["c_type"] for p in param_specs)
        function_prototype = f"{return_type} {function_name}({proto_params});"
        if extra_source:
            function_prototype = extra_source

        source = HARNESS_TEMPLATE.format(
            function_prototype=function_prototype,
            param_declarations=param_decls,
            return_type=return_type,
            function_call=function_call,
            property_check=property_check or "/* no property check */",
            declare_defaults=declare_defaults,
            pass_args=pass_args,
        )

        # Write source
        src_path = self.output_dir / f"{name}.c"
        src_path.write_text(source, encoding="utf-8")

        # Compile
        bin_path = self.output_dir / name
        cmd = [self.cc] + self.cflags + ["-o", str(bin_path), str(src_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")

        # Build symbol specs for verification request
        symbols = []
        for p in param_specs:
            symbols.append({
                "name": p["name"],
                "bits": p.get("bits", 32),
                "kind": p.get("kind", "int"),
                "length": p.get("length"),
            })

        return HarnessTarget(
            binary_name=name,
            binary_path=str(bin_path),
            entry="target_wrapper",
            symbols=symbols,
        )
