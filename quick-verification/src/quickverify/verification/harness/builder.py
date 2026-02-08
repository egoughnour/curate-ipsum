"""HarnessBuilder — transforms function specs into compiled harness binaries.

Generates a minimal C harness wrapping a target function, compiles it with
debug flags, and returns metadata for the angr verification adapter.
"""
from __future__ import annotations

import hashlib
import logging
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class HarnessTarget:
    """Output of a successful harness build."""
    binary_name: str
    binary_path: str
    entry: str
    symbols: List[Dict[str, Any]]
    source_path: str = ""


# Jinja-style template (using Python .format)
BASIC_HARNESS = """\
/* Auto-generated harness — do not edit */
#include <stdint.h>
#include <stdlib.h>

__attribute__((noinline)) void violation(void) {{
    volatile int *p = (int*)0;
    *p = 1;
}}

__attribute__((noinline)) void ok_exit(void) {{ return; }}

{extra_source}

__attribute__((noinline)) {return_type} target_wrapper({param_decls}) {{
    {return_type} result = {function_call};
    {property_check}
    return result;
}}

int main(int argc, char **argv) {{
    {declare_defaults}
    (void)target_wrapper({pass_args});
    ok_exit();
    return 0;
}}
"""


class HarnessBuilder:
    """Builds compiled C harness binaries for verification.

    Usage::

        builder = HarnessBuilder(output_dir="artifacts/harnesses")
        target = builder.build(
            function_name="target_fn",
            param_specs=[{"name": "x", "c_type": "int32_t", "bits": 32, "kind": "int"}],
            property_check='if (result == 1337) violation();',
        )
        # target.binary_path → "artifacts/harnesses/target_fn_harness"
    """

    DEFAULT_CFLAGS = ["-O0", "-g", "-fno-omit-frame-pointer", "-fno-inline"]

    def __init__(
        self,
        cc: str = "gcc",
        cflags: Optional[List[str]] = None,
        output_dir: str = "artifacts/harnesses",
    ):
        self.cc = cc
        self.cflags = cflags or list(self.DEFAULT_CFLAGS)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        function_name: str,
        param_specs: List[Dict[str, Any]],
        property_check: str = "/* no property check */",
        return_type: str = "int",
        extra_source: str = "",
        harness_name: Optional[str] = None,
    ) -> HarnessTarget:
        """Generate, compile, and return a harness target.

        Args:
            function_name: Name of the function to wrap.
            param_specs: List of ``{"name", "c_type", "bits", "kind"}``.
            property_check: C code for the property check body.
            return_type: Return type of the target function.
            extra_source: Additional C source (e.g., the target function body).
            harness_name: Output binary name.
        """
        name = harness_name or f"{function_name}_harness"

        param_decls = ", ".join(f"{p['c_type']} {p['name']}" for p in param_specs)
        pass_args = ", ".join(p["name"] for p in param_specs)
        declare_defaults = "\n    ".join(
            f"{p['c_type']} {p['name']} = 0;" for p in param_specs
        )
        function_call = f"{function_name}({pass_args})"

        if not extra_source:
            proto_params = ", ".join(p["c_type"] for p in param_specs)
            extra_source = f"extern {return_type} {function_name}({proto_params});"

        source = BASIC_HARNESS.format(
            extra_source=extra_source,
            return_type=return_type,
            param_decls=param_decls or "void",
            function_call=function_call,
            property_check=property_check,
            declare_defaults=declare_defaults or "/* no params */",
            pass_args=pass_args,
        )

        src_path = self.output_dir / f"{name}.c"
        src_path.write_text(source, encoding="utf-8")

        bin_path = self.output_dir / name
        cmd = [self.cc, *self.cflags, "-o", str(bin_path), str(src_path)]
        log.info("Compiling: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed ({self.cc}):\n{result.stderr}")

        symbols = [
            {
                "name": p["name"],
                "bits": p.get("bits", 32),
                "kind": p.get("kind", "int"),
                "length": p.get("length"),
            }
            for p in param_specs
        ]

        return HarnessTarget(
            binary_name=name,
            binary_path=str(bin_path.resolve()),
            entry="target_wrapper",
            symbols=symbols,
            source_path=str(src_path.resolve()),
        )

    def build_from_spec(self, spec: Any) -> HarnessTarget:
        """Build from a PropertySpec (orchestrator integration).

        Extracts function_name and symbols from the spec. Requires
        spec to have ``target_hint``, ``symbols``, and optionally
        ``find_value`` for property check generation.
        """
        function_name = getattr(spec, "target_hint", None) or "target_fn"
        symbols = getattr(spec, "symbols", None) or []
        find_value = getattr(spec, "find_value", "violation")

        param_specs = []
        for s in symbols:
            c_type_map = {"int": f"int{s.bits}_t", "bool": "uint8_t", "bytes": "uint8_t*"}
            param_specs.append({
                "name": s.name,
                "c_type": c_type_map.get(s.kind, f"int{s.bits}_t"),
                "bits": s.bits,
                "kind": s.kind,
            })

        property_check = f'if (result == 1337) violation();  /* placeholder */'

        return self.build(
            function_name=function_name,
            param_specs=param_specs,
            property_check=property_check,
        )
