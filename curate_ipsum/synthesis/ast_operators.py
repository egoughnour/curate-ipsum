"""
AST-aware genetic operators for code synthesis.

Crossover: swap compatible subtrees between two parent ASTs.
Mutation: directed modifications guided by counterexample analysis.

All operators validate output via ast.parse() — invalid results are discarded.
"""

from __future__ import annotations

import ast
import copy
import logging
import random
from typing import Any

from curate_ipsum.synthesis.models import Counterexample, Individual, PatchSource

LOG = logging.getLogger("synthesis.ast_operators")


class ASTCrossover:
    """AST-aware crossover between parent patches."""

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int = 0,
    ) -> tuple[Individual | None, Individual | None]:
        """
        Swap compatible subtrees between two parents.

        Returns two children, or (None, None) if crossover fails.
        Both children are validated for syntactic correctness.
        """
        try:
            tree1 = ast.parse(parent1.code)
            tree2 = ast.parse(parent2.code)
        except SyntaxError:
            return None, None

        # Find compatible subtree pairs
        pairs = self._find_compatible_subtrees(tree1, tree2)
        if not pairs:
            return None, None

        # Pick a random compatible pair
        (node1, parent_node1, field1, idx1), (node2, parent_node2, field2, idx2) = random.choice(pairs)

        # Create deep copies for children
        child_tree1 = copy.deepcopy(tree1)
        child_tree2 = copy.deepcopy(tree2)

        # Perform the swap on copies
        try:
            self._swap_nodes(
                child_tree1, child_tree2, node1, node2, parent_node1, parent_node2, field1, field2, idx1, idx2
            )
        except Exception:
            return None, None

        # Unparse and validate
        child1 = self._tree_to_individual(child_tree1, [parent1.id, parent2.id], generation)
        child2 = self._tree_to_individual(child_tree2, [parent2.id, parent1.id], generation)

        return child1, child2

    def _find_compatible_subtrees(self, tree1: ast.AST, tree2: ast.AST) -> list[tuple[Any, ...]]:
        """Find pairs of subtrees with the same node type at comparable depth."""
        nodes1 = self._collect_swappable_nodes(tree1)
        nodes2 = self._collect_swappable_nodes(tree2)

        pairs = []
        for info1 in nodes1:
            for info2 in nodes2:
                if type(info1[0]) is type(info2[0]):
                    pairs.append((info1, info2))

        return pairs

    def _collect_swappable_nodes(self, tree: ast.AST) -> list[tuple[Any, ...]]:
        """Collect nodes that can be swapped (statements and expressions)."""
        swappable = []

        for parent_node in ast.walk(tree):
            for field_name, value in ast.iter_fields(parent_node):
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, (ast.stmt, ast.expr)):
                            swappable.append((item, parent_node, field_name, idx))
                elif isinstance(value, (ast.stmt, ast.expr)):
                    swappable.append((value, parent_node, field_name, -1))

        return swappable

    def _swap_nodes(self, tree1, tree2, node1, node2, parent1, parent2, field1, field2, idx1, idx2) -> None:
        """Swap two nodes between trees (operates on deep copies)."""
        # Find corresponding nodes in the copies by position
        copy_nodes1 = self._collect_swappable_nodes(tree1)
        copy_nodes2 = self._collect_swappable_nodes(tree2)

        if not copy_nodes1 or not copy_nodes2:
            return

        # Use first available pair from copies
        cn1 = copy_nodes1[0]
        cn2 = copy_nodes2[0]

        # Simple swap: replace first swappable node in each tree
        p1, f1, i1 = cn1[1], cn1[2], cn1[3]
        p2, f2, i2 = cn2[1], cn2[2], cn2[3]

        val1 = getattr(p1, f1)
        val2 = getattr(p2, f2)

        if isinstance(val1, list) and isinstance(val2, list) and i1 >= 0 and i2 >= 0:
            if i1 < len(val1) and i2 < len(val2):
                val1[i1], val2[i2] = val2[i2], val1[i1]

    @staticmethod
    def _tree_to_individual(
        tree: ast.AST,
        lineage: list[str],
        generation: int,
    ) -> Individual | None:
        """Convert AST back to Individual, returning None if invalid."""
        try:
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
            ind = Individual(
                code=code,
                lineage=lineage,
                generation=generation,
                source=PatchSource.CROSSOVER,
            )
            return ind if ind.is_valid() else None
        except Exception:
            return None


class ASTMutator:
    """
    Directed mutation operators guided by counterexample analysis.

    Operators:
    - constant_tweak: modify numeric/string constants
    - operator_swap: replace +/- with -/+, </> with >/< etc.
    - guard_insertion: add if-checks for edge cases
    - branch_flip: swap if/else branches
    - argument_reorder: shuffle function arguments
    """

    OPERATORS = ["constant_tweak", "operator_swap", "guard_insertion", "branch_flip"]

    def mutate(
        self,
        individual: Individual,
        generation: int = 0,
        counterexample: Counterexample | None = None,
    ) -> Individual | None:
        """
        Apply a single mutation operator.

        If a counterexample is provided, select the most relevant operator.
        Otherwise, pick randomly.
        """
        try:
            tree = ast.parse(individual.code)
        except SyntaxError:
            return None

        if counterexample:
            operator = self._select_operator_for_ce(counterexample)
        else:
            operator = random.choice(self.OPERATORS)

        mutated_tree = copy.deepcopy(tree)

        try:
            if operator == "constant_tweak":
                self._apply_constant_tweak(mutated_tree)
            elif operator == "operator_swap":
                self._apply_operator_swap(mutated_tree)
            elif operator == "guard_insertion":
                self._apply_guard_insertion(mutated_tree)
            elif operator == "branch_flip":
                self._apply_branch_flip(mutated_tree)
        except Exception:
            return None

        try:
            ast.fix_missing_locations(mutated_tree)
            code = ast.unparse(mutated_tree)
            ind = Individual(
                code=code,
                lineage=[individual.id],
                generation=generation,
                source=PatchSource.MUTATION,
                metadata={"mutation_operator": operator},
            )
            return ind if ind.is_valid() else None
        except Exception:
            return None

    def _select_operator_for_ce(self, ce: Counterexample) -> str:
        """Map counterexample error type to the most relevant mutation operator."""
        msg = ce.error_message.lower()
        if "type" in msg or "typeerror" in msg:
            return "guard_insertion"  # Add type checks
        if "index" in msg or "range" in msg or "bound" in msg:
            return "guard_insertion"  # Add bounds checks
        if "assert" in msg or "expected" in msg:
            return "constant_tweak"  # Adjust values
        if "wrong" in msg or "incorrect" in msg:
            return "operator_swap"  # Try different operators
        return random.choice(self.OPERATORS)

    def _apply_constant_tweak(self, tree: ast.AST) -> None:
        """Modify a random numeric constant."""
        constants = [
            node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
        ]
        if constants:
            target = random.choice(constants)
            if isinstance(target.value, int):
                target.value = target.value + random.choice([-1, 1, -2, 2])
            elif isinstance(target.value, float):
                target.value = target.value * random.uniform(0.8, 1.2)

    def _apply_operator_swap(self, tree: ast.AST) -> None:
        """Swap a comparison or binary operator."""
        COMP_SWAPS = {
            ast.Lt: ast.LtE,
            ast.LtE: ast.Lt,
            ast.Gt: ast.GtE,
            ast.GtE: ast.Gt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
        }
        BIN_SWAPS = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.FloorDiv,
            ast.FloorDiv: ast.Mult,
        }

        # Try comparison operators first
        compares = [n for n in ast.walk(tree) if isinstance(n, ast.Compare) and n.ops]
        if compares:
            target = random.choice(compares)
            idx = random.randrange(len(target.ops))
            old_op = type(target.ops[idx])
            if old_op in COMP_SWAPS:
                target.ops[idx] = COMP_SWAPS[old_op]()
                return

        # Try binary operators
        binops = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)]
        if binops:
            target = random.choice(binops)
            old_op = type(target.op)
            if old_op in BIN_SWAPS:
                target.op = BIN_SWAPS[old_op]()

    def _apply_guard_insertion(self, tree: ast.AST) -> None:
        """Insert a guard clause (if check) at the beginning of a function."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                # Add: if arg is None: return None (for first arg)
                if node.args.args:
                    arg_name = node.args.args[0].arg
                    guard = ast.If(
                        test=ast.Compare(
                            left=ast.Name(id=arg_name, ctx=ast.Load()),
                            ops=[ast.Is()],
                            comparators=[ast.Constant(value=None)],
                        ),
                        body=[ast.Return(value=ast.Constant(value=None))],
                        orelse=[],
                    )
                    node.body.insert(0, guard)
                    return

    def _apply_branch_flip(self, tree: ast.AST) -> None:
        """Swap if/else branches."""
        ifs = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.orelse]
        if ifs:
            target = random.choice(ifs)
            target.body, target.orelse = target.orelse, target.body
