"""
Entropy-aware diversity maintenance for the genetic algorithm.

Monitors Shannon entropy of the population's structural features.
When entropy drops below threshold (premature convergence), injects
diversity by requesting novel candidates from the LLM client.

No sklearn dependency — uses simple binning for clustering.
"""

from __future__ import annotations

import ast
import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from synthesis.models import Individual, Specification, SynthesisConfig

LOG = logging.getLogger("synthesis.entropy")


class EntropyManager:
    """Monitor and maintain population diversity."""

    def __init__(self, config: SynthesisConfig) -> None:
        self._config = config

    def compute_entropy(self, individuals: List[Individual]) -> float:
        """
        Compute Shannon entropy over structural feature clusters.

        High entropy = diverse population (good).
        Low entropy = convergence, possibly premature (needs injection).

        Returns entropy in bits. Max = log2(n) for n individuals.
        """
        if len(individuals) <= 1:
            return 0.0

        features = [self._extract_features(ind) for ind in individuals]
        clusters = [self._feature_to_bin(f) for f in features]

        cluster_counts = Counter(clusters)
        total = len(individuals)

        entropy = 0.0
        for count in cluster_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def needs_injection(self, individuals: List[Individual]) -> bool:
        """Check if population entropy is below threshold."""
        entropy = self.compute_entropy(individuals)
        return entropy < self._config.entropy_threshold

    def select_for_replacement(
        self,
        individuals: List[Individual],
        n: int,
    ) -> List[int]:
        """
        Select indices of the n most similar individuals for replacement.

        Strategy: find the most common feature bin, select the n lowest-fitness
        individuals from that bin.
        """
        features = [self._extract_features(ind) for ind in individuals]
        bins = [self._feature_to_bin(f) for f in features]

        # Find most common (over-represented) bin
        bin_counts = Counter(bins)
        most_common_bin = bin_counts.most_common(1)[0][0]

        # Collect indices in that bin, sorted by fitness (ascending)
        candidates = [
            (i, individuals[i].fitness)
            for i, b in enumerate(bins)
            if b == most_common_bin
        ]
        candidates.sort(key=lambda x: x[1])

        return [idx for idx, _ in candidates[:n]]

    def _extract_features(self, individual: Individual) -> Dict[str, float]:
        """
        Extract structural features from code for diversity measurement.

        Features:
        - ast_depth: max nesting depth
        - node_count: total AST nodes
        - branch_count: number of if/for/while
        - func_count: number of function definitions
        - var_count: number of unique variable names
        """
        try:
            tree = ast.parse(individual.code)
        except SyntaxError:
            return {"ast_depth": 0, "node_count": 0, "branch_count": 0,
                    "func_count": 0, "var_count": 0}

        node_count = sum(1 for _ in ast.walk(tree))
        branch_count = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.If, ast.For, ast.While))
        )
        func_count = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        var_names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                var_names.add(n.id)
        ast_depth = self._max_depth(tree)

        return {
            "ast_depth": float(ast_depth),
            "node_count": float(node_count),
            "branch_count": float(branch_count),
            "func_count": float(func_count),
            "var_count": float(len(var_names)),
        }

    def _max_depth(self, node: ast.AST, current: int = 0) -> int:
        """Compute maximum nesting depth of an AST."""
        max_child = current
        for child in ast.iter_child_nodes(node):
            child_depth = self._max_depth(child, current + 1)
            if child_depth > max_child:
                max_child = child_depth
        return max_child

    @staticmethod
    def _feature_to_bin(features: Dict[str, float]) -> str:
        """
        Map features to a discrete bin for entropy computation.

        Simple binning: quantize each feature to a small number of levels.
        """
        def _bin(val: float, boundaries: List[float]) -> int:
            for i, b in enumerate(boundaries):
                if val <= b:
                    return i
            return len(boundaries)

        depth_bin = _bin(features.get("ast_depth", 0), [2, 4, 6, 8])
        nodes_bin = _bin(features.get("node_count", 0), [10, 25, 50, 100])
        branch_bin = _bin(features.get("branch_count", 0), [1, 3, 5])
        var_bin = _bin(features.get("var_count", 0), [3, 6, 10])

        return f"{depth_bin}-{nodes_bin}-{branch_bin}-{var_bin}"
