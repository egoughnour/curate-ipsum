# LPython + KLEE Integration Feasibility Assessment

**Date**: 2026-01-27
**Purpose**: Evaluate using LPython (or alternatives) for unified AST extraction + symbolic execution pipeline

## Executive Summary

The proposal to use LPython for both M2 (graph infrastructure) and symbolic execution is **architecturally sound** but requires a **phased approach** due to LPython's alpha status. A hybrid strategy is recommended: use libasr for AST/call-graph extraction now, while the LLVM backend matures.

---

## The Vision

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Single Semantic Representation                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Python Source                                                         │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     ┌─────────┐                                          │
│   │   AST   │────▶│   ASR   │◀── Abstract Semantic Representation      │
│   └─────────┘     └────┬────┘                                          │
│                        │                                                │
│           ┌────────────┼────────────┐                                  │
│           │            │            │                                  │
│           ▼            ▼            ▼                                  │
│      ┌────────┐   ┌────────┐   ┌────────┐                              │
│      │  M2    │   │   C    │   │  LLVM  │                              │
│      │ Graph  │   │ Output │   │   IR   │                              │
│      └────────┘   └───┬────┘   └────────┘                              │
│           │           │                                                 │
│           ▼           ▼                                                 │
│      Call Graph   ┌────────┐                                           │
│      Extraction   │ Clang  │                                           │
│                   └───┬────┘                                           │
│                       ▼                                                 │
│                  ┌─────────┐                                           │
│                  │  LLVM   │                                           │
│                  │ Bitcode │                                           │
│                  └────┬────┘                                           │
│                       │                                                 │
│                       ▼                                                 │
│                  ┌─────────┐                                           │
│                  │  KLEE   │                                           │
│                  └────┬────┘                                           │
│                       │                                                 │
│                       ▼                                                 │
│                  Path Coverage                                          │
│                  + Test Cases                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Same ASR serves both call-graph analysis AND compilation, ensuring semantic consistency.

---

## Tool Comparison

### LPython

| Aspect | Status | Notes |
|--------|--------|-------|
| AST Extraction | ✅ Ready | Standalone module, designed for tooling |
| ASR (Semantic) | ✅ Ready | Shared via libasr, type-aware |
| C Backend | ✅ Ready | Production-quality |
| C++ Backend | ⚠️ Partial | Under development |
| LLVM Backend | ❌ JIT only | Direct LLVM not yet stable |
| Python Coverage | ⚠️ Subset | Type-annotated, numerical focus |
| Call Graph | 🔧 DIY | Extract from ASR (not built-in) |

**Path to KLEE**: Python → ASR → C → Clang → LLVM Bitcode → KLEE

### Codon

| Aspect | Status | Notes |
|--------|--------|-------|
| AST Extraction | ⚠️ Internal | Not designed for external tooling |
| LLVM Backend | ✅ Ready | Direct compilation |
| NumPy Support | ✅ Native | Full reimplementation |
| Python Coverage | ⚠️ Subset | Broader than LPython |
| Call Graph | ❌ None | Would need custom IR walker |

**Path to KLEE**: Python → Codon → LLVM Bitcode → KLEE (direct, but less flexible)

### Alternative: pyan + Separate Compiler

| Aspect | Status | Notes |
|--------|--------|-------|
| Call Graph | ✅ Ready | pyan3, PyCG available |
| Semantic Consistency | ❌ None | Different AST than compiler |
| Maintenance | ⚠️ Orphaned | pyan seeking new maintainer |
| Python Coverage | ✅ Full | Standard AST module |

---

## libasr Architecture Deep-Dive

The [libasr](https://github.com/lcompilers/libasr) repository is the key enabler:

```
libasr/
├── ASR/                    # Abstract Semantic Representation
│   ├── asr.h              # Core ASR nodes
│   ├── asr_utils.h        # Traversal utilities
│   └── asr_verify.cpp     # Validation
├── backends/
│   ├── c.cpp              # C code generation
│   ├── cpp.cpp            # C++ code generation
│   ├── llvm.cpp           # LLVM IR generation
│   └── wasm.cpp           # WebAssembly
├── passes/                 # ASR → ASR transformations
│   ├── inline_functions.cpp
│   ├── dead_code_removal.cpp
│   ├── loop_unrolling.cpp
│   └── ...
└── codegen/               # Backend utilities
```

### ASR Node Types (relevant for call graphs)

```cpp
// From asr.h (conceptual)
enum class exprType {
    FunctionCall,      // Direct call
    SubroutineCall,    // Void function call
    ClassMethod,       // Method call
    ...
};

struct Function {
    string name;
    vector<arg> args;
    vector<stmt> body;
    expr return_type;
    ...
};
```

### Call Graph Extraction Strategy

```python
# Pseudo-code for ASR call graph extraction
def extract_call_graph(asr_module):
    graph = DiGraph()

    for func in asr_module.functions:
        graph.add_node(func.name, kind="function")

        for stmt in walk_asr(func.body):
            if isinstance(stmt, FunctionCall):
                graph.add_edge(func.name, stmt.target_name)
            elif isinstance(stmt, ClassMethod):
                graph.add_edge(func.name, f"{stmt.class_name}.{stmt.method_name}")

    return graph
```

---

## KLEE Integration Workflow

### Step 1: Compile to C (via LPython)

```bash
# LPython generates C code
lpython --show-c input.py -o input.c
```

### Step 2: Add Symbolic Annotations

```c
// Before (LPython output)
int compute(int x, int y) {
    if (x > y) return x - y;
    else return y - x;
}

// After (annotated for KLEE)
#include <klee/klee.h>

int compute(int x, int y) {
    klee_make_symbolic(&x, sizeof(x), "x");
    klee_make_symbolic(&y, sizeof(y), "y");

    if (x > y) return x - y;
    else return y - x;
}
```

### Step 3: Compile to LLVM Bitcode

```bash
clang -I /path/to/klee/include -emit-llvm -c -g -O0 input.c -o input.bc
```

### Step 4: Run KLEE

```bash
klee --output-dir=klee-out input.bc
```

### Step 5: Parse Path Coverage

```python
def parse_klee_output(klee_dir: Path) -> PathCoverage:
    """Parse KLEE output for path coverage data."""
    coverage = PathCoverage()

    # Parse run.stats for execution statistics
    stats_file = klee_dir / "run.stats"
    if stats_file.exists():
        coverage.stats = parse_klee_stats(stats_file)

    # Parse test cases (*.ktest files)
    for ktest in klee_dir.glob("*.ktest"):
        test_case = parse_ktest(ktest)
        coverage.test_cases.append(test_case)

    # Parse path conditions from info files
    for info in klee_dir.glob("test*.info"):
        path = parse_path_info(info)
        coverage.paths.append(path)

    return coverage
```

---

## Code Map Requirements

For KLEE work, we need a mapping between:
- Original Python source locations
- Generated C source locations
- LLVM IR locations (debug info)
- KLEE path constraints

```python
@dataclass
class CodeMap:
    """Maps between Python, C, and LLVM locations."""

    # Python source → C source
    py_to_c: Dict[PyLocation, CLocation]

    # C source → LLVM debug info
    c_to_llvm: Dict[CLocation, LLVMDebugLoc]

    # KLEE path → Python regions
    def klee_path_to_regions(self, path: KleePath) -> List[Region]:
        """Map KLEE execution path back to Python regions."""
        regions = []
        for constraint in path.constraints:
            llvm_loc = constraint.location
            c_loc = self.llvm_to_c(llvm_loc)
            py_loc = self.c_to_py(c_loc)
            regions.append(py_loc.to_region())
        return regions
```

### LPython Source Maps

LPython's C backend can emit location comments:

```c
// Generated by LPython
// Source: input.py:42
int result = x + y;  // input.py:42
```

We would need to:
1. Parse these comments during C generation
2. Preserve them through Clang compilation (via `-g`)
3. Map back from KLEE's LLVM locations

---

## Implementation Phases

### Phase 1: ASR-Based Call Graph (M2 Foundation)

**Timeline**: 2-3 weeks
**Dependencies**: libasr, networkx

```python
# curate_ipsum/graph/asr_extractor.py

from pathlib import Path
from typing import Dict, List, Set
import subprocess
import json

class ASRCallGraphExtractor:
    """Extract call graphs from Python via LPython ASR."""

    def __init__(self, lpython_path: str = "lpython"):
        self.lpython = lpython_path

    def extract(self, source_files: List[Path]) -> "CallGraph":
        """Extract call graph from Python source files."""
        # Step 1: Generate ASR JSON via LPython
        asr_json = self._generate_asr(source_files)

        # Step 2: Walk ASR to find function definitions and calls
        functions = self._extract_functions(asr_json)
        calls = self._extract_calls(asr_json)

        # Step 3: Build graph
        return self._build_graph(functions, calls)

    def _generate_asr(self, files: List[Path]) -> dict:
        """Run LPython to generate ASR."""
        result = subprocess.run(
            [self.lpython, "--show-asr", "--json"] + [str(f) for f in files],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
```

### Phase 2: C Generation Pipeline

**Timeline**: 1-2 weeks
**Dependencies**: LPython C backend

```python
# curate_ipsum/symbolic/lpython_bridge.py

class LPythonBridge:
    """Bridge between curate-ipsum and LPython compilation."""

    def compile_to_c(
        self,
        source: Path,
        output: Path,
        emit_source_map: bool = True,
    ) -> CompilationResult:
        """Compile Python to C via LPython."""
        args = [self.lpython, "--show-c", str(source), "-o", str(output)]

        result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode != 0:
            raise CompilationError(result.stderr)

        source_map = None
        if emit_source_map:
            source_map = self._parse_source_comments(output)

        return CompilationResult(
            c_source=output,
            source_map=source_map,
            warnings=self._parse_warnings(result.stderr),
        )
```

### Phase 3: Symbolic Annotation

**Timeline**: 1 week
**Dependencies**: KLEE headers

```python
# curate_ipsum/symbolic/annotator.py

class SymbolicAnnotator:
    """Annotate C code with KLEE symbolic markers."""

    KLEE_INCLUDE = '#include <klee/klee.h>'

    def annotate(
        self,
        c_source: Path,
        symbolic_vars: List[SymbolicVar],
        output: Path,
    ) -> None:
        """Add klee_make_symbolic calls for specified variables."""
        source = c_source.read_text()

        # Add KLEE include
        if self.KLEE_INCLUDE not in source:
            source = self.KLEE_INCLUDE + "\n\n" + source

        # Find entry point and add symbolic annotations
        for var in symbolic_vars:
            annotation = self._make_symbolic(var)
            source = self._inject_after_declaration(source, var.name, annotation)

        output.write_text(source)

    def _make_symbolic(self, var: SymbolicVar) -> str:
        return f'klee_make_symbolic(&{var.name}, sizeof({var.name}), "{var.name}");'
```

### Phase 4: KLEE Execution & Parsing

**Timeline**: 1-2 weeks
**Dependencies**: KLEE, Clang

```python
# curate_ipsum/symbolic/klee_runner.py

class KLEERunner:
    """Run KLEE symbolic execution and parse results."""

    def run(
        self,
        bitcode: Path,
        output_dir: Path,
        timeout: int = 3600,
        max_memory: int = 8192,
    ) -> KLEEResult:
        """Execute KLEE on LLVM bitcode."""
        args = [
            "klee",
            f"--output-dir={output_dir}",
            f"--max-time={timeout}",
            f"--max-memory={max_memory}",
            "--emit-all-errors",
            "--write-paths",
            str(bitcode),
        ]

        result = subprocess.run(args, capture_output=True, text=True)

        return KLEEResult(
            stats=self._parse_stats(output_dir / "run.stats"),
            tests=self._parse_tests(output_dir),
            paths=self._parse_paths(output_dir),
            errors=self._parse_errors(output_dir),
        )
```

### Phase 5: Integration with curate-ipsum

**Timeline**: 1 week
**Dependencies**: Phases 1-4

```python
# curate_ipsum/symbolic/__init__.py

class SymbolicExecutionPipeline:
    """End-to-end symbolic execution for Python code."""

    def __init__(self):
        self.lpython = LPythonBridge()
        self.annotator = SymbolicAnnotator()
        self.klee = KLEERunner()
        self.graph_extractor = ASRCallGraphExtractor()

    async def analyze(
        self,
        source: Path,
        entry_points: List[str],
        symbolic_inputs: List[SymbolicVar],
    ) -> SymbolicAnalysisResult:
        """Full symbolic execution pipeline."""

        # 1. Extract call graph (for M2)
        call_graph = self.graph_extractor.extract([source])

        # 2. Compile to C
        c_result = self.lpython.compile_to_c(source, self.work_dir / "output.c")

        # 3. Annotate symbolics
        annotated = self.work_dir / "annotated.c"
        self.annotator.annotate(c_result.c_source, symbolic_inputs, annotated)

        # 4. Compile to bitcode
        bitcode = self._compile_to_bitcode(annotated)

        # 5. Run KLEE
        klee_result = self.klee.run(bitcode, self.work_dir / "klee-out")

        # 6. Map results back to Python
        coverage = self._map_coverage(klee_result, c_result.source_map)

        return SymbolicAnalysisResult(
            call_graph=call_graph,
            path_coverage=coverage,
            test_cases=klee_result.tests,
            source_map=c_result.source_map,
        )
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LPython subset too narrow | Medium | High | Start with type-annotated code; contribute upstream |
| LLVM version mismatch | Low | Medium | Pin LLVM versions across toolchain |
| Source map accuracy | Medium | Medium | Add verification tests; compare AST locations |
| KLEE scalability | Medium | High | Limit symbolic scope; use compositional analysis |
| libasr API changes | Low | Medium | Pin version; abstract behind interface |

---

## Alternatives Considered

### Mypyc + KLEE

Mypyc compiles type-annotated Python to C extensions, but:
- Generates CPython API calls (not pure C)
- Would need significant modification for KLEE compatibility
- No ASR for call graph extraction

### Cython + KLEE

Cython is mature but:
- Generates CPython-dependent code
- Would require heavy transformation for KLEE
- Call graph extraction would be separate

### angr (instead of KLEE)

angr is Python-native symbolic execution:
- Works on binaries (no source needed)
- More flexible but slower
- Could be fallback if KLEE integration proves difficult

---

## Recommendation

**Proceed with LPython-based approach** using this sequence:

1. **Now**: Implement ASR-based call graph extraction (M2)
2. **Parallel**: Monitor LPython LLVM backend progress
3. **Q2**: Implement C → KLEE pipeline
4. **Q3**: Integrate symbolic results with belief revision (M3)

The unified ASR approach provides long-term value even if we need to use the C backend (via Clang) initially instead of direct LLVM output.

---

## Next Steps

1. [ ] Install LPython and verify `--show-asr` output format
2. [ ] Prototype call graph extraction from ASR JSON
3. [ ] Test C generation on representative Python code
4. [ ] Set up KLEE development environment
5. [ ] Create end-to-end proof-of-concept
