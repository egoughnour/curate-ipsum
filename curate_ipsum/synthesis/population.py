"""
Population management for the genetic algorithm.

Handles individual selection, replacement, and population-level operations.
The population is the mutable state of the GA loop — it evolves across iterations.
"""

from __future__ import annotations

import logging
import random

from curate_ipsum.synthesis.models import Individual, PatchSource

LOG = logging.getLogger("synthesis.population")


class Population:
    """Manages a population of candidate patches for genetic evolution."""

    def __init__(self, individuals: list[Individual] | None = None) -> None:
        self._individuals: list[Individual] = list(individuals or [])

    @classmethod
    def from_candidates(
        cls,
        candidates: list[str],
        generation: int = 0,
        source: PatchSource = PatchSource.LLM,
    ) -> "Population":
        """Initialize population from raw code strings (e.g., LLM outputs)."""
        individuals = []
        for code in candidates:
            ind = Individual(
                code=code,
                generation=generation,
                source=source,
            )
            if ind.is_valid():
                individuals.append(ind)
            else:
                LOG.debug("Discarded syntactically invalid candidate: %.60s...", code)
        LOG.info(
            "Initialized population: %d valid / %d total candidates",
            len(individuals),
            len(candidates),
        )
        return cls(individuals)

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self):
        return iter(self._individuals)

    @property
    def individuals(self) -> list[Individual]:
        return list(self._individuals)

    @property
    def best(self) -> Individual | None:
        """Return the individual with highest fitness, or None if empty."""
        if not self._individuals:
            return None
        return max(self._individuals, key=lambda ind: ind.fitness)

    @property
    def average_fitness(self) -> float:
        if not self._individuals:
            return 0.0
        return sum(ind.fitness for ind in self._individuals) / len(self._individuals)

    def select_elite(self, n: int) -> list[Individual]:
        """Select top-n individuals by fitness."""
        n = min(n, len(self._individuals))
        sorted_pop = sorted(self._individuals, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:n]

    def tournament_select(self, n: int, k: int = 3) -> list[Individual]:
        """
        Select n individuals via k-tournament selection.

        For each selection: pick k random individuals, keep the fittest.
        """
        if len(self._individuals) < k:
            return list(self._individuals)

        selected: list[Individual] = []
        for _ in range(n):
            competitors = random.sample(self._individuals, min(k, len(self._individuals)))
            winner = max(competitors, key=lambda ind: ind.fitness)
            selected.append(winner)
        return selected

    def add_individual(self, individual: Individual) -> None:
        self._individuals.append(individual)

    def add_individuals(self, individuals: list[Individual]) -> None:
        self._individuals.extend(individuals)

    def remove_weakest(self, n: int) -> list[Individual]:
        """Remove and return the n weakest individuals."""
        n = min(n, len(self._individuals))
        sorted_pop = sorted(self._individuals, key=lambda ind: ind.fitness)
        removed = sorted_pop[:n]
        remaining = sorted_pop[n:]
        self._individuals = remaining
        return removed

    def replace_with(self, new_generation: list[Individual]) -> None:
        """Replace entire population with a new generation."""
        self._individuals = list(new_generation)

    def size(self) -> int:
        return len(self._individuals)
