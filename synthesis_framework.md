# Synthesis Framework: From LLM Candidates to Strongly-Typed Patches

## The Core Problem

LLMs produce **plausible** code, not **provably correct** code. Their outputs are:
- Statistically likely given training distribution
- Syntactically valid (usually)
- Semantically approximate (often)
- Type-correct only by accident
- Behaviorally equivalent rarely

**The Solution**: Treat LLM outputs as **seed populations** for formal synthesis, not as final artifacts.

## Multi-Framework Mutation Architecture

### Unified Interface Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    Curate-Ipsum Unified Interface                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐ │
│  │ Stryker │  │ mutmut  │  │cosmic-  │  │ poodle  │  │univ-  │ │
│  │  (JS)   │  │  (Py)   │  │  ray    │  │  (Py)   │  │mutator│ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘ │
│       │            │            │            │           │      │
│       └────────────┴─────┬──────┴────────────┴───────────┘      │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │  Region Assignment    │                          │
│              │  Engine (implicit     │                          │
│              │  detection via graph  │                          │
│              │  spectral analysis)   │                          │
│              └───────────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

### Implicit Region Detection

Regions that "no one else is likely to ever detect" emerge from:

```python
class ImplicitRegionDetector:
    """Detect code regions through graph-spectral analysis."""

    def detect_orphan_regions(self, graph: CodeGraph) -> List[Region]:
        """Regions with low centrality but high complexity."""
        return [
            node for node in graph.nodes
            if self.centrality(node) < 0.2
            and self.cyclomatic_complexity(node) > 10
            and self.test_coverage(node) < 0.5
        ]

    def detect_bridge_regions(self, graph: CodeGraph) -> List[Region]:
        """Regions that are graph cut vertices - removal disconnects modules."""
        return list(nx.articulation_points(graph))

    def detect_spectral_anomalies(self, graph: CodeGraph) -> List[Region]:
        """Regions near Fiedler partition boundaries with high local variance."""
        fiedler = self.compute_fiedler_vector(graph)
        # Nodes where neighbors have very different Fiedler values
        return [
            node for node in graph.nodes
            if np.std([fiedler[n] for n in graph.neighbors(node)]) > threshold
        ]

    def detect_mutation_resistant(self, history: RunHistory) -> List[Region]:
        """Regions where mutants consistently survive across frameworks."""
        # Cross-framework surviving mutant intersection
        survivors_by_framework = {
            fw: set(get_survivors(history, fw))
            for fw in ['stryker', 'mutmut', 'cosmic-ray']
        }
        # Intersection = resistant to all frameworks
        return set.intersection(*survivors_by_framework.values())
```

### Non-Contradictory Framework Orchestration

```python
class FrameworkOrchestrator:
    """Coordinate multiple mutation frameworks without contradiction."""

    def partition_codebase(self, graph: CodeGraph) -> Dict[str, List[Region]]:
        """Assign regions to frameworks based on strengths."""

        assignments = defaultdict(list)

        for region in graph.regions:
            if region.language == 'javascript':
                assignments['stryker'].append(region)
            elif region.is_highly_dynamic:
                assignments['mutmut'].append(region)  # Better with dynamic Python
            elif region.needs_custom_operators:
                assignments['cosmic-ray'].append(region)  # Plugin support
            elif region.is_simple:
                assignments['poodle'].append(region)  # Fast for simple code
            else:
                assignments['universalmutator'].append(region)  # Fallback

        # Verify no contradictions
        self._verify_disjoint(assignments)
        return assignments

    def _verify_disjoint(self, assignments: Dict[str, List[Region]]):
        """Ensure no region assigned to contradictory frameworks."""
        all_regions = []
        for regions in assignments.values():
            for r in regions:
                if r in all_regions:
                    raise ConflictError(f"Region {r} assigned to multiple frameworks")
                all_regions.append(r)
```

## CEGIS: Counterexample-Guided Inductive Synthesis

### The CEGIS Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                         CEGIS Loop                               │
│                                                                  │
│   ┌──────────┐    candidate    ┌──────────┐                     │
│   │ Synthesizer│─────────────→│ Verifier │                      │
│   │ (LLM +    │               │ (SMT/    │                      │
│   │  genetic) │←─────────────│  KLEE)   │                      │
│   └──────────┘  counterexample └──────────┘                     │
│        ▲                            │                            │
│        │                            │ verified                   │
│        │                            ▼                            │
│   ┌──────────┐               ┌──────────┐                       │
│   │ LLM      │               │ Strongly │                       │
│   │ Candidates│              │ Typed    │                       │
│   │ (top k)  │               │ Patch    │                       │
│   └──────────┘               └──────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class CEGISEngine:
    """Counterexample-Guided Inductive Synthesis for code patches."""

    def __init__(self, verifier: SMTVerifier, synthesizer: HybridSynthesizer):
        self.verifier = verifier
        self.synthesizer = synthesizer
        self.counterexamples: List[Counterexample] = []

    def synthesize(
        self,
        spec: Specification,
        llm_candidates: List[CodePatch],
        max_iterations: int = 100
    ) -> Optional[StronglyTypedPatch]:
        """
        CEGIS loop: refine candidates until verified or exhausted.
        """
        # Initialize population with LLM candidates
        population = self.synthesizer.initialize(llm_candidates)

        for iteration in range(max_iterations):
            # Select best candidate from population
            candidate = self.synthesizer.select_best(population, self.counterexamples)

            # Verify candidate against specification
            result = self.verifier.verify(candidate, spec)

            if result.verified:
                return self._to_strongly_typed(candidate, result.proof)

            # Extract counterexample
            counterexample = result.counterexample
            self.counterexamples.append(counterexample)

            # Refine population using counterexample
            population = self.synthesizer.refine(
                population,
                counterexample,
                spec
            )

        return None  # Synthesis failed

    def _to_strongly_typed(
        self,
        candidate: CodePatch,
        proof: VerificationProof
    ) -> StronglyTypedPatch:
        """Convert verified candidate to strongly-typed patch with proof."""
        return StronglyTypedPatch(
            code=candidate.code,
            type_signature=self._infer_types(candidate),
            preconditions=proof.preconditions,
            postconditions=proof.postconditions,
            invariants=proof.invariants,
            proof_certificate=proof.certificate
        )
```

## CEGAR: Counterexample-Guided Abstraction Refinement

### Abstraction Hierarchy

```
Concrete Program
      ↓ (abstract)
Abstract Model (coarse)
      ↓ (verify)
Spurious Counterexample?
      ↓ (refine if spurious)
Abstract Model (finer)
      ↓ (repeat until real CE or verified)
Verified / Real Counterexample
```

### Implementation

```python
class CEGAREngine:
    """Counterexample-Guided Abstraction Refinement for patch verification."""

    def __init__(self):
        self.abstraction_levels: List[AbstractionLevel] = [
            TypeAbstraction(),      # Coarsest: just types
            ControlFlowAbstraction(),  # CFG structure
            DataFlowAbstraction(),     # DFG with abstract values
            ConcreteExecution(),       # Finest: actual execution
        ]

    def verify(self, patch: CodePatch, spec: Specification) -> VerificationResult:
        """CEGAR loop: start coarse, refine on spurious counterexamples."""

        current_level = 0

        while current_level < len(self.abstraction_levels):
            abstraction = self.abstraction_levels[current_level]
            abstract_model = abstraction.abstract(patch)

            result = self._verify_abstract(abstract_model, spec)

            if result.verified:
                return VerificationResult(verified=True, level=current_level)

            if self._is_real_counterexample(result.counterexample, patch):
                return VerificationResult(
                    verified=False,
                    counterexample=result.counterexample
                )

            # Spurious counterexample - refine abstraction
            current_level += 1

        # Reached concrete level
        return self._concrete_verify(patch, spec)

    def _is_real_counterexample(
        self,
        ce: AbstractCounterexample,
        patch: CodePatch
    ) -> bool:
        """Check if abstract counterexample is realizable in concrete program."""
        # Attempt to concretize the counterexample
        concrete_ce = self._concretize(ce)
        if concrete_ce is None:
            return False  # Spurious

        # Execute concrete counterexample
        return self._execute_and_check(concrete_ce, patch)
```

## Genetic Algorithm Population Management

### Epoch-Based Evolution

```python
@dataclass
class Individual:
    """A candidate patch in the population."""
    code: str
    fitness: float
    lineage: List[str]  # Parent IDs for tracking
    mutations_applied: List[MutationOperator]
    generation: int

class GeneticSynthesizer:
    """Genetic algorithm for patch population evolution."""

    def __init__(self, config: GAConfig):
        self.population_size = config.population_size
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate
        self.elite_ratio = config.elite_ratio
        self.entropy_threshold = config.entropy_threshold

    def evolve_epoch(
        self,
        population: List[Individual],
        counterexamples: List[Counterexample],
        spec: Specification
    ) -> List[Individual]:
        """Evolve population for one epoch."""

        # Evaluate fitness against counterexamples
        for individual in population:
            individual.fitness = self._evaluate_fitness(
                individual, counterexamples, spec
            )

        # Selection: tournament + elitism
        elite = self._select_elite(population)
        parents = self._tournament_select(population)

        # Crossover: combine successful patches
        offspring = self._crossover(parents)

        # Mutation: directed by counterexamples
        mutated = self._mutate_directed(offspring, counterexamples)

        # Maintain entropy (diversity)
        if self._population_entropy(mutated) < self.entropy_threshold:
            mutated = self._inject_diversity(mutated)

        return elite + mutated

    def _evaluate_fitness(
        self,
        individual: Individual,
        counterexamples: List[Counterexample],
        spec: Specification
    ) -> float:
        """
        Fitness = (counterexamples avoided) + (spec conditions met) - (complexity)
        """
        ce_score = sum(
            1 for ce in counterexamples
            if not self._triggers_counterexample(individual, ce)
        ) / len(counterexamples) if counterexamples else 1.0

        spec_score = self._spec_satisfaction_ratio(individual, spec)

        complexity_penalty = self._ast_complexity(individual.code) / 100

        return (0.4 * ce_score) + (0.5 * spec_score) - (0.1 * complexity_penalty)

    def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """AST-aware crossover between parent patches."""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]

            if random.random() < self.crossover_rate:
                # AST subtree exchange
                child1, child2 = self._ast_crossover(p1, p2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([p1, p2])

        return offspring

    def _mutate_directed(
        self,
        population: List[Individual],
        counterexamples: List[Counterexample]
    ) -> List[Individual]:
        """Apply mutations directed by counterexample analysis."""
        mutated = []

        for individual in population:
            if random.random() < self.mutation_rate:
                # Analyze which counterexample is closest to being avoided
                closest_ce = self._find_closest_counterexample(
                    individual, counterexamples
                )

                # Apply mutation that addresses the counterexample
                mutation_op = self._select_mutation_for_ce(closest_ce)
                new_individual = self._apply_mutation(individual, mutation_op)
                mutated.append(new_individual)
            else:
                mutated.append(individual)

        return mutated
```

### Entropy-Aware Diversity Maintenance

```python
class EntropyManager:
    """Maintain population diversity through entropy monitoring."""

    def compute_population_entropy(self, population: List[Individual]) -> float:
        """
        Compute Shannon entropy over population's structural features.
        High entropy = diverse population
        Low entropy = convergence (potentially premature)
        """
        # Extract structural features
        features = [self._extract_features(ind) for ind in population]

        # Cluster features
        clusters = self._cluster_features(features)

        # Compute entropy over cluster distribution
        cluster_counts = Counter(clusters)
        total = len(population)

        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in cluster_counts.values()
            if count > 0
        )

        return entropy

    def inject_diversity(
        self,
        population: List[Individual],
        target_entropy: float
    ) -> List[Individual]:
        """Inject diverse individuals to reach target entropy."""

        current_entropy = self.compute_population_entropy(population)

        while current_entropy < target_entropy:
            # Generate structurally novel individual
            novel = self._generate_novel_individual(population)
            population = self._replace_most_similar(population, novel)
            current_entropy = self.compute_population_entropy(population)

        return population

    def _generate_novel_individual(
        self,
        population: List[Individual]
    ) -> Individual:
        """Generate individual maximally different from current population."""
        # Compute centroid of current population in feature space
        features = [self._extract_features(ind) for ind in population]
        centroid = np.mean(features, axis=0)

        # Generate in opposite direction from centroid
        direction = -centroid / np.linalg.norm(centroid)

        # Convert feature direction to code generation constraints
        constraints = self._features_to_constraints(direction)

        # Generate novel code satisfying anti-centroid constraints
        return self._constrained_generation(constraints)
```

## LLM Integration: Top-K Candidates as Seed Population

### The Pipeline

```
User Query / Failing Test / Surviving Mutant
                    ↓
        ┌──────────────────────┐
        │   LLM (Claude/GPT)   │
        │   Generate top-k     │
        │   candidate patches  │
        └──────────────────────┘
                    ↓
         [k candidates, unverified]
                    ↓
        ┌──────────────────────┐
        │  Initial Population  │
        │  (seed from LLM)     │
        └──────────────────────┘
                    ↓
        ┌──────────────────────┐
        │  CEGIS + CEGAR +     │
        │  Genetic Evolution   │
        │  (intense analysis)  │
        └──────────────────────┘
                    ↓
        ┌──────────────────────┐
        │  Strongly Typed      │
        │  Verified Patch      │
        │  (with proof)        │
        └──────────────────────┘
```

### Implementation

```python
class LLMGuidedSynthesis:
    """Use LLM candidates as seeds for formal synthesis."""

    def __init__(
        self,
        llm_client: LLMClient,
        cegis: CEGISEngine,
        cegar: CEGAREngine,
        genetic: GeneticSynthesizer
    ):
        self.llm = llm_client
        self.cegis = cegis
        self.cegar = cegar
        self.genetic = genetic

    async def synthesize_patch(
        self,
        context: SynthesisContext,
        k: int = 10
    ) -> Optional[StronglyTypedPatch]:
        """
        Full pipeline: LLM → genetic → CEGIS → strongly typed.
        """
        # Step 1: Get top-k candidates from LLM
        llm_candidates = await self._get_llm_candidates(context, k)

        # Step 2: Initialize genetic population with LLM candidates
        population = self.genetic.initialize_from_candidates(llm_candidates)

        # Step 3: Extract specification from context
        spec = self._extract_specification(context)

        # Step 4: CEGIS loop with genetic refinement
        max_epochs = 50
        counterexamples = []

        for epoch in range(max_epochs):
            # Evolve population
            population = self.genetic.evolve_epoch(
                population, counterexamples, spec
            )

            # Get best candidate
            best = self.genetic.select_best(population)

            # Verify with CEGAR
            result = self.cegar.verify(best.code, spec)

            if result.verified:
                return self._finalize_patch(best, result)

            # Add counterexample for next epoch
            counterexamples.append(result.counterexample)

            # Log progress
            self._log_epoch(epoch, best.fitness, len(counterexamples))

        return None  # Synthesis failed

    async def _get_llm_candidates(
        self,
        context: SynthesisContext,
        k: int
    ) -> List[CodePatch]:
        """Query LLM for k candidate patches."""

        prompt = self._build_synthesis_prompt(context)

        # Request multiple completions
        responses = await self.llm.complete(
            prompt=prompt,
            n=k,
            temperature=0.8,  # Higher temp for diversity
            max_tokens=2000
        )

        # Parse code from responses
        candidates = []
        for response in responses:
            code = self._extract_code(response)
            if code and self._syntactically_valid(code):
                candidates.append(CodePatch(code=code, source='llm'))

        return candidates
```

## Strongly Typed Patch Output

### The Final Artifact

```python
@dataclass
class StronglyTypedPatch:
    """A verified, formally typed patch with proof certificate."""

    # The code itself
    code: str
    diff: str  # Unified diff format

    # Type information
    type_signature: TypeSignature
    generic_parameters: List[TypeVar]
    type_constraints: List[TypeConstraint]

    # Formal verification
    preconditions: List[Predicate]
    postconditions: List[Predicate]
    invariants: List[Predicate]
    proof_certificate: ProofCertificate

    # Provenance
    synthesis_method: Literal['cegis', 'cegar', 'genetic', 'hybrid']
    llm_seed_id: Optional[str]
    counterexamples_resolved: int
    epochs_required: int
    verification_time_ms: int

    # Confidence metrics
    entropy_at_synthesis: float
    fitness_score: float
    abstraction_level_verified: int

    def to_pr_description(self) -> str:
        """Generate PR description with verification details."""
        return f"""
## Strongly Typed Patch

### Type Signature
```
{self.type_signature}
```

### Verification
- **Method**: {self.synthesis_method}
- **Counterexamples resolved**: {self.counterexamples_resolved}
- **Abstraction level**: {self.abstraction_level_verified}
- **Proof certificate**: {self.proof_certificate.hash[:16]}...

### Preconditions
{self._format_predicates(self.preconditions)}

### Postconditions
{self._format_predicates(self.postconditions)}

### Confidence
- Entropy at synthesis: {self.entropy_at_synthesis:.3f}
- Fitness score: {self.fitness_score:.3f}
"""
```

## The Fundamental Insight

> "Only by using calculation-intense and entropy-aware methods can we hope to steer the models we are using into actual structured output of the strongly typed variety."

This captures the essential asymmetry:

| LLM Outputs | Formal Synthesis |
|-------------|-----------------|
| O(1) generation | O(2^n) worst case |
| Probabilistic | Deterministic |
| Plausible | Provable |
| Weakly typed | Strongly typed |
| No guarantees | Formal guarantees |

**The synthesis framework bridges this gap**: use LLMs for cheap candidate generation, then invest computational resources to achieve formal guarantees. The genetic algorithm maintains population diversity (entropy-aware) while CEGIS/CEGAR provide the verification backbone.

This is the only path to **trustworthy AI-generated code**.
