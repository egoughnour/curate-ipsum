# Belief Revision Framework: Totalizing Theories with Provenance

## The Core Insight

Failed patches aren't failures - they're **evidence**. Each rolled-back assertion sharpens the universal model by eliminating regions of the hypothesis space. The synthesis process becomes a belief revision process where:

- **Expansion**: New LLM candidates add beliefs
- **Contraction**: Counterexamples remove beliefs
- **Revision**: Contradictions trigger minimal belief change

## AGM Postulates Applied to Patch Synthesis

The AGM framework (Alchourrón, Gärdenfors, Makinson 1985) provides axioms for rational belief revision. Applied to code synthesis:

### Classical AGM Postulates (Adapted)

| AGM Postulate | Synthesis Interpretation |
|---------------|-------------------------|
| **Closure** | Theory closed under logical consequence (type inference) |
| **Success** | New evidence always incorporated |
| **Inclusion** | Revision ⊆ Expansion (don't add beliefs beyond evidence) |
| **Vacuity** | If ¬φ ∉ K, then K+φ = K*φ (consistent additions are simple) |
| **Consistency** | K*φ is consistent if φ is consistent |
| **Extensionality** | Logically equivalent inputs yield same revision |

### The Totalizing Theory

```python
@dataclass
class SynthesisTheory:
    """A totalizing theory about correct patches with full provenance."""

    # Core beliefs about what constitutes correct code
    type_beliefs: Set[TypeAssertion]
    behavior_beliefs: Set[BehaviorAssertion]
    invariant_beliefs: Set[InvariantAssertion]

    # Grounding: every belief traced to evidence
    provenance: Dict[Assertion, List[Evidence]]

    # Revision history for rollback
    revision_log: List[RevisionEvent]

    # Entrenchment ordering (which beliefs to sacrifice first)
    entrenchment: Dict[Assertion, float]

    def expand(self, assertion: Assertion, evidence: Evidence) -> 'SynthesisTheory':
        """AGM expansion: add belief when consistent."""
        if self._is_consistent_with(assertion):
            new_theory = self._copy()
            new_theory._add_belief(assertion, evidence)
            new_theory.revision_log.append(
                RevisionEvent(type='expand', assertion=assertion, evidence=evidence)
            )
            return new_theory
        else:
            # Must revise instead
            return self.revise(assertion, evidence)

    def contract(self, assertion: Assertion, evidence: Evidence) -> 'SynthesisTheory':
        """AGM contraction: remove belief minimally."""
        if assertion not in self._all_beliefs():
            return self  # Nothing to contract

        new_theory = self._copy()

        # Use entrenchment to guide minimal contraction
        to_remove = new_theory._compute_contraction_set(assertion)
        for belief in to_remove:
            new_theory._remove_belief(belief)

        new_theory.revision_log.append(
            RevisionEvent(type='contract', assertion=assertion, evidence=evidence)
        )
        return new_theory

    def revise(self, assertion: Assertion, evidence: Evidence) -> 'SynthesisTheory':
        """AGM revision: contract ¬assertion, then expand with assertion."""
        # Levi identity: K*φ = (K÷¬φ)+φ
        negation = self._negate(assertion)
        contracted = self.contract(negation, evidence)
        revised = contracted.expand(assertion, evidence)
        return revised

    def rollback(self, to_event: RevisionEvent) -> 'SynthesisTheory':
        """Rollback to a previous theory state."""
        idx = self.revision_log.index(to_event)
        # Replay log up to that point
        new_theory = SynthesisTheory.empty()
        for event in self.revision_log[:idx + 1]:
            new_theory = new_theory._apply_event(event)
        return new_theory
```

## Grounded Assertions

Every assertion must be grounded in evidence. Ungrounded assertions are inadmissible.

### Evidence Types

```python
class EvidenceType(Enum):
    TEST_PASS = "test_pass"           # Test execution succeeded
    TEST_FAIL = "test_fail"           # Test execution failed
    TYPE_CHECK = "type_check"         # Type checker accepted/rejected
    SMT_SAT = "smt_sat"              # SMT solver found satisfying assignment
    SMT_UNSAT = "smt_unsat"          # SMT solver proved unsatisfiable
    COUNTEREXAMPLE = "counterexample" # Concrete counterexample found
    PROOF = "proof"                   # Formal proof constructed
    MUTATION_KILLED = "mutation_killed"
    MUTATION_SURVIVED = "mutation_survived"
    LLM_SUGGESTION = "llm_suggestion" # Weak evidence, low entrenchment

@dataclass
class Evidence:
    """Grounding for an assertion."""
    type: EvidenceType
    source: str                    # Where this evidence came from
    timestamp: datetime
    reproducible: bool             # Can we re-verify this?
    confidence: float              # 0.0 to 1.0
    raw_data: Dict[str, Any]       # The actual evidence (logs, proofs, etc.)

    def strength(self) -> float:
        """Evidence strength determines entrenchment of derived beliefs."""
        base_strength = {
            EvidenceType.PROOF: 1.0,
            EvidenceType.SMT_UNSAT: 0.95,
            EvidenceType.SMT_SAT: 0.9,
            EvidenceType.COUNTEREXAMPLE: 0.85,
            EvidenceType.TYPE_CHECK: 0.8,
            EvidenceType.TEST_PASS: 0.7,
            EvidenceType.TEST_FAIL: 0.7,
            EvidenceType.MUTATION_KILLED: 0.6,
            EvidenceType.MUTATION_SURVIVED: 0.5,
            EvidenceType.LLM_SUGGESTION: 0.2,  # Weak!
        }
        return base_strength[self.type] * self.confidence
```

### Grounding Rules

```python
class GroundingChecker:
    """Ensure all assertions are properly grounded."""

    def check_grounding(self, theory: SynthesisTheory) -> List[GroundingViolation]:
        """Verify all beliefs have adequate grounding."""
        violations = []

        for assertion in theory.all_beliefs():
            evidence_list = theory.provenance.get(assertion, [])

            if not evidence_list:
                violations.append(GroundingViolation(
                    assertion=assertion,
                    reason="No evidence provided"
                ))
                continue

            # Check evidence strength
            max_strength = max(e.strength() for e in evidence_list)
            if max_strength < 0.5:
                violations.append(GroundingViolation(
                    assertion=assertion,
                    reason=f"Weak evidence (max strength {max_strength:.2f})"
                ))

            # Check reproducibility for strong claims
            if assertion.is_strong_claim():
                if not any(e.reproducible for e in evidence_list):
                    violations.append(GroundingViolation(
                        assertion=assertion,
                        reason="Strong claim requires reproducible evidence"
                    ))

        return violations
```

## Entrenchment Ordering

Not all beliefs are equal. When revision requires contraction, we sacrifice less-entrenched beliefs first.

### Entrenchment Factors

```python
class EntrenchmentCalculator:
    """Compute entrenchment ordering for beliefs."""

    def compute_entrenchment(
        self,
        assertion: Assertion,
        theory: SynthesisTheory
    ) -> float:
        """
        Entrenchment based on:
        1. Evidence strength
        2. Logical dependencies (core beliefs are more entrenched)
        3. Age (older beliefs slightly more entrenched)
        4. Corroboration (multiple independent evidence sources)
        """
        evidence_list = theory.provenance.get(assertion, [])

        # Factor 1: Evidence strength
        evidence_score = max(
            (e.strength() for e in evidence_list),
            default=0.0
        )

        # Factor 2: Logical centrality
        dependents = theory.beliefs_depending_on(assertion)
        centrality_score = min(len(dependents) / 10, 1.0)

        # Factor 3: Age (normalized by log)
        if evidence_list:
            oldest = min(e.timestamp for e in evidence_list)
            age_days = (datetime.now() - oldest).days
            age_score = min(math.log(age_days + 1) / 5, 1.0)
        else:
            age_score = 0.0

        # Factor 4: Corroboration
        independent_sources = len(set(e.source for e in evidence_list))
        corroboration_score = min(independent_sources / 3, 1.0)

        # Weighted combination
        entrenchment = (
            0.4 * evidence_score +
            0.3 * centrality_score +
            0.1 * age_score +
            0.2 * corroboration_score
        )

        return entrenchment
```

### Minimal Contraction via Entrenchment

```python
def compute_contraction_set(
    self,
    assertion: Assertion,
    theory: SynthesisTheory
) -> Set[Assertion]:
    """
    Find minimal set of beliefs to remove to eliminate assertion.
    Uses entrenchment to prefer removing less-entrenched beliefs.
    """
    # Find all ways to derive the assertion
    derivation_paths = theory.find_derivation_paths(assertion)

    if not derivation_paths:
        # Assertion is basic belief, remove directly
        return {assertion}

    # For each path, we must remove at least one belief
    # Choose the belief with lowest entrenchment from each path
    candidates = []
    for path in derivation_paths:
        min_entrenched = min(
            path,
            key=lambda b: theory.entrenchment[b]
        )
        candidates.append(min_entrenched)

    # Hitting set problem: find minimal set that blocks all paths
    return self._minimal_hitting_set(derivation_paths, theory.entrenchment)
```

## Failure Mode Reasoning

By tracking provenance and revision history, we can reason about *why* patches fail.

### Failure Mode Taxonomy

```python
class FailureMode(Enum):
    TYPE_MISMATCH = "type_mismatch"
    PRECONDITION_VIOLATION = "precondition_violation"
    POSTCONDITION_VIOLATION = "postcondition_violation"
    INVARIANT_VIOLATION = "invariant_violation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SEMANTIC_DRIFT = "semantic_drift"  # Correct but wrong behavior
    OVERFITTING = "overfitting"        # Passes tests but wrong in general
    UNDERFITTING = "underfitting"      # Too general, misses requirements

@dataclass
class FailureAnalysis:
    """Analysis of why a patch candidate failed."""
    mode: FailureMode
    root_cause: Assertion           # The belief that was violated
    evidence: Evidence              # The evidence that contradicted it
    causal_chain: List[Assertion]   # How we got here
    suggested_revision: Optional[Assertion]  # What to believe instead
```

### Failure Mode Detection

```python
class FailureModeAnalyzer:
    """Analyze patch failures to guide revision."""

    def analyze_failure(
        self,
        patch: CodePatch,
        error: VerificationError,
        theory: SynthesisTheory
    ) -> FailureAnalysis:
        """Determine failure mode and suggest revision."""

        if isinstance(error, TypeError):
            return self._analyze_type_failure(patch, error, theory)
        elif isinstance(error, CounterexampleError):
            return self._analyze_counterexample(patch, error, theory)
        elif isinstance(error, TimeoutError):
            return self._analyze_timeout(patch, error, theory)
        # ... etc

    def _analyze_counterexample(
        self,
        patch: CodePatch,
        error: CounterexampleError,
        theory: SynthesisTheory
    ) -> FailureAnalysis:
        """Analyze why counterexample defeats patch."""

        ce = error.counterexample

        # Find which belief the counterexample violates
        violated_beliefs = []
        for belief in theory.behavior_beliefs:
            if not belief.holds_for(ce):
                violated_beliefs.append(belief)

        if not violated_beliefs:
            # Counterexample reveals missing belief
            mode = FailureMode.UNDERFITTING
            suggested = self._infer_missing_belief(ce, patch)
        else:
            # Existing belief was wrong
            mode = FailureMode.POSTCONDITION_VIOLATION
            # Suggest weakening the violated belief
            suggested = self._weaken_belief(violated_beliefs[0], ce)

        return FailureAnalysis(
            mode=mode,
            root_cause=violated_beliefs[0] if violated_beliefs else None,
            evidence=Evidence(
                type=EvidenceType.COUNTEREXAMPLE,
                source='smt_solver',
                timestamp=datetime.now(),
                reproducible=True,
                confidence=0.95,
                raw_data={'counterexample': ce}
            ),
            causal_chain=self._trace_causal_chain(violated_beliefs, theory),
            suggested_revision=suggested
        )
```

## Totalizing Theory Construction

The goal is to form a **complete** theory about correct patches - one that can answer any well-formed question.

### Theory Completion

```python
class TheoryCompletion:
    """Drive theory toward totalization."""

    def identify_gaps(self, theory: SynthesisTheory) -> List[TheoryGap]:
        """Find questions the theory cannot answer."""
        gaps = []

        # Type completeness: can we type all expressions?
        for expr in theory.relevant_expressions():
            if not theory.can_type(expr):
                gaps.append(TheoryGap(
                    type='type_gap',
                    subject=expr,
                    question="What is the type of this expression?"
                ))

        # Behavioral completeness: defined on all inputs?
        for func in theory.relevant_functions():
            uncovered_inputs = theory.find_uncovered_inputs(func)
            if uncovered_inputs:
                gaps.append(TheoryGap(
                    type='behavioral_gap',
                    subject=func,
                    question=f"What is the behavior on inputs {uncovered_inputs}?"
                ))

        # Invariant completeness: all loops have invariants?
        for loop in theory.relevant_loops():
            if not theory.has_invariant(loop):
                gaps.append(TheoryGap(
                    type='invariant_gap',
                    subject=loop,
                    question="What invariant holds for this loop?"
                ))

        return gaps

    def propose_completion(
        self,
        gap: TheoryGap,
        theory: SynthesisTheory
    ) -> List[Assertion]:
        """Propose assertions to fill a gap."""

        if gap.type == 'type_gap':
            # Use type inference to propose
            return self._infer_types(gap.subject, theory)

        elif gap.type == 'behavioral_gap':
            # Use symbolic execution to explore behavior
            return self._explore_behavior(gap.subject, theory)

        elif gap.type == 'invariant_gap':
            # Use Houdini-style invariant inference
            return self._infer_invariants(gap.subject, theory)

        return []
```

### Aggressive Totalization Strategy

```python
class AggressiveTotalizer:
    """
    Aggressively form complete theories, accepting rollback as learning.
    """

    def totalize(
        self,
        theory: SynthesisTheory,
        candidates: List[CodePatch]
    ) -> SynthesisTheory:
        """
        Aggressively add beliefs until we have a total theory,
        then refine via counterexample-driven rollback.
        """
        current = theory

        # Phase 1: Aggressive expansion
        for candidate in candidates:
            assertions = self._extract_assertions(candidate)
            for assertion in assertions:
                # Add even weakly-grounded beliefs
                evidence = Evidence(
                    type=EvidenceType.LLM_SUGGESTION,
                    source=f'candidate_{candidate.id}',
                    timestamp=datetime.now(),
                    reproducible=False,
                    confidence=0.3,
                    raw_data={'candidate': candidate}
                )
                current = current.expand(assertion, evidence)

        # Phase 2: Identify and fill gaps
        gaps = self.identify_gaps(current)
        for gap in gaps:
            proposals = self.propose_completion(gap, current)
            for proposal in proposals:
                # Speculatively add proposals
                evidence = Evidence(
                    type=EvidenceType.LLM_SUGGESTION,  # or inference
                    source='gap_completion',
                    timestamp=datetime.now(),
                    reproducible=True,
                    confidence=0.5,
                    raw_data={'gap': gap}
                )
                current = current.expand(proposal, evidence)

        # Phase 3: Verify and contract on failure
        counterexamples = self._find_counterexamples(current)
        for ce in counterexamples:
            # Each counterexample triggers contraction
            violated = self._find_violated_assertion(ce, current)
            evidence = Evidence(
                type=EvidenceType.COUNTEREXAMPLE,
                source='verification',
                timestamp=datetime.now(),
                reproducible=True,
                confidence=0.95,
                raw_data={'counterexample': ce}
            )
            current = current.contract(violated, evidence)

        return current
```

## Provenance DAG

The revision history forms a DAG that enables sophisticated reasoning about belief evolution.

```
            ┌─────────────────────────────────────────┐
            │           Provenance DAG                │
            │                                         │
            │    [Initial Theory T₀]                  │
            │           │                             │
            │    expand(type_assertion_1)             │
            │           │                             │
            │         [T₁]                            │
            │         / \                             │
            │   expand   revise                       │
            │       /       \                         │
            │    [T₂]      [T₃]                       │
            │      │         │                        │
            │  contract   expand                      │
            │      │         │                        │
            │    [T₄]      [T₅]                       │
            │        \     /                          │
            │         merge                           │
            │           │                             │
            │         [T₆]                            │
            │           │                             │
            │    (current theory)                     │
            └─────────────────────────────────────────┘
```

### Provenance Queries

```python
class ProvenanceQuery:
    """Query the provenance DAG for belief history."""

    def why_believe(
        self,
        assertion: Assertion,
        theory: SynthesisTheory
    ) -> List[Evidence]:
        """Trace why we believe this assertion."""
        return theory.provenance.get(assertion, [])

    def when_added(
        self,
        assertion: Assertion,
        theory: SynthesisTheory
    ) -> Optional[RevisionEvent]:
        """Find when this belief was added."""
        for event in theory.revision_log:
            if event.type == 'expand' and event.assertion == assertion:
                return event
        return None

    def counterfactual(
        self,
        assertion: Assertion,
        theory: SynthesisTheory
    ) -> SynthesisTheory:
        """What would theory look like without this belief?"""
        # Find when assertion was added
        add_event = self.when_added(assertion, theory)
        if add_event is None:
            return theory

        # Rollback to just before
        return theory.rollback_to_before(add_event)

    def belief_stability(
        self,
        assertion: Assertion,
        theory: SynthesisTheory
    ) -> float:
        """How stable has this belief been? (0 = constantly revised, 1 = stable)"""
        events_involving = [
            e for e in theory.revision_log
            if e.assertion == assertion
        ]

        if not events_involving:
            return 1.0

        # Count how many times it was added vs removed
        adds = sum(1 for e in events_involving if e.type == 'expand')
        removes = sum(1 for e in events_involving if e.type == 'contract')

        if adds == 0:
            return 0.0

        return 1.0 - (removes / adds)
```

## The Universal Model

The synthesis process produces not just patches, but a **universal model** of what correct code looks like for this specification.

```python
@dataclass
class UniversalModel:
    """The totalized theory as a universal model of correctness."""

    # The current theory state
    theory: SynthesisTheory

    # Invariants that held across all revisions
    stable_invariants: Set[Assertion]

    # Assertions that were revised multiple times (uncertain)
    unstable_assertions: Set[Assertion]

    # The "core" - beliefs that were never contracted
    core_beliefs: Set[Assertion]

    # Learned negative knowledge (what correct code is NOT)
    negative_knowledge: Set[Assertion]

    def validity_certificate(self) -> ValidityCertificate:
        """Generate certificate of model validity."""
        return ValidityCertificate(
            core_beliefs=self.core_beliefs,
            stable_invariants=self.stable_invariants,
            grounding_check=self._verify_all_grounded(),
            consistency_check=self._verify_consistency(),
            completeness_gaps=self._identify_remaining_gaps()
        )

    def extract_patch(self) -> StronglyTypedPatch:
        """Extract the best patch from the universal model."""
        # The patch is the unique program satisfying all core beliefs
        return self._synthesize_from_beliefs(self.core_beliefs)
```

## Summary: Rollback Sharpens Validity

> "As long as every assertion is grounded somehow, rolling it back only sharpens the validity of a universal model."

This is the key insight:

1. **Aggressive expansion** fills the theory with candidates
2. **Grounding requirements** ensure every belief has evidence
3. **Counterexamples trigger contraction** - removing weakly-entrenched beliefs
4. **Entrenchment ordering** ensures we keep the most-supported beliefs
5. **Provenance tracking** lets us understand why the model evolved
6. **Rollback as refinement** - each failure makes the remaining theory more valid

The universal model that survives this process is **refined by failure**, not just constructed from success.
