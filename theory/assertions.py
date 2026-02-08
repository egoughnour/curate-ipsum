"""
Typed assertion model for the belief revision engine.

Assertions are typed beliefs about code properties. Each assertion is
grounded by evidence (test results, mutation results, etc.) and tracked
with confidence scores.

Assertion types:
    - TYPE: "variable x has type int"
    - BEHAVIOR: "function f returns positive values for positive inputs"
    - INVARIANT: "loop counter i is always < len(items)"
    - CONTRACT: "function f satisfies precondition P and postcondition Q"
    - PRECONDITION: "x > 0 holds before calling f"
    - POSTCONDITION: "result != None holds after calling f"
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AssertionKind(str, Enum):
    """Classification of code assertions."""

    TYPE = "type"
    BEHAVIOR = "behavior"
    INVARIANT = "invariant"
    CONTRACT = "contract"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"


@dataclass
class Assertion:
    """
    A typed belief about code properties.

    Assertions are the atomic units of belief in the synthesis theory.
    Each must be grounded by at least one piece of evidence.
    """

    id: str
    kind: AssertionKind
    content: str
    confidence: float  # 0.0 to 1.0
    region_id: Optional[str] = None
    grounding_evidence_ids: List[str] = field(default_factory=list)
    created_utc: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if isinstance(self.kind, str):
            self.kind = AssertionKind(self.kind)


def assertion_to_node_dict(assertion: Assertion) -> Dict[str, Any]:
    """
    Serialize an Assertion to a CASStore-compatible node dict.

    Args:
        assertion: The assertion to serialize

    Returns:
        Dict suitable for storing as a Node in CASStore
    """
    return {
        "id": assertion.id,
        "domain_id": "code_mutation",
        "kind": "Assertion",
        "properties": {
            "assertion_type": assertion.kind.value,
            "content": assertion.content,
            "confidence": assertion.confidence,
            "region_id": assertion.region_id,
            "grounding_evidence_ids": assertion.grounding_evidence_ids,
            **assertion.metadata,
        },
        "created_utc": assertion.created_utc,
    }


def node_dict_to_assertion(node: Dict[str, Any]) -> Assertion:
    """
    Deserialize a CASStore node dict to an Assertion.

    Args:
        node: Dict from CASStore with Node kind

    Returns:
        Assertion instance

    Raises:
        ValueError: If node is not a valid Assertion
    """
    props = node.get("properties", {})
    kind_str = props.get("assertion_type")

    if kind_str is None:
        raise ValueError(f"Node {node.get('id')} has no assertion_type property")

    return Assertion(
        id=node["id"],
        kind=AssertionKind(kind_str),
        content=props.get("content", ""),
        confidence=props.get("confidence", 0.5),
        region_id=props.get("region_id"),
        grounding_evidence_ids=props.get("grounding_evidence_ids", []),
        created_utc=node.get("created_utc", ""),
        metadata={
            k: v
            for k, v in props.items()
            if k
            not in (
                "assertion_type",
                "content",
                "confidence",
                "region_id",
                "grounding_evidence_ids",
            )
        },
    )


class ContradictionDetector:
    """
    Detects logical conflicts between assertions.

    Contradiction detection is essential for AGM revision — when a new
    assertion contradicts existing beliefs, the contradicted beliefs must
    be contracted before the new assertion can be added.

    Contradiction rules:
    1. Same-region TYPE conflict: two TYPE assertions for the same region
       with different content (e.g., "x: int" vs "x: str")
    2. Negated BEHAVIOR: a BEHAVIOR assertion that logically negates another
       for the same region (detected by keyword heuristics)
    3. CONTRACT violations: a PRECONDITION that contradicts a POSTCONDITION
       or vice versa within the same region
    """

    # Negation keywords that suggest logical opposition
    _NEGATION_PAIRS = [
        ("always", "never"),
        ("true", "false"),
        ("positive", "negative"),
        ("greater", "less"),
        ("increases", "decreases"),
        ("returns", "does not return"),
        ("handles", "does not handle"),
        ("accepts", "rejects"),
        ("allows", "disallows"),
        ("valid", "invalid"),
        ("none", "not none"),
        ("null", "not null"),
        ("empty", "not empty"),
    ]

    @classmethod
    def find_contradictions(
        cls,
        new_assertion: Assertion,
        existing_assertions: List[Assertion],
    ) -> List[Assertion]:
        """
        Find assertions that contradict the new one.

        Args:
            new_assertion: The assertion being added
            existing_assertions: All existing assertions in the theory

        Returns:
            List of existing assertions that conflict with the new one
        """
        contradictions = []

        for existing in existing_assertions:
            if cls._contradicts(new_assertion, existing):
                contradictions.append(existing)

        return contradictions

    @classmethod
    def _contradicts(cls, a: Assertion, b: Assertion) -> bool:
        """Check if two assertions contradict each other."""
        # Must be in the same region to conflict
        if a.region_id != b.region_id or a.region_id is None:
            return False

        # Same ID means same assertion — not a contradiction
        if a.id == b.id:
            return False

        # Type conflicts: two TYPE assertions for same region with different content
        if a.kind == AssertionKind.TYPE and b.kind == AssertionKind.TYPE:
            return a.content.strip().lower() != b.content.strip().lower()

        # Behavior negation: check for negation keywords
        if a.kind == AssertionKind.BEHAVIOR and b.kind == AssertionKind.BEHAVIOR:
            return cls._is_negated(a.content, b.content)

        # Contract conflicts: pre/postcondition inconsistencies
        if (
            a.kind == AssertionKind.PRECONDITION
            and b.kind == AssertionKind.PRECONDITION
        ):
            return cls._is_negated(a.content, b.content)

        if (
            a.kind == AssertionKind.POSTCONDITION
            and b.kind == AssertionKind.POSTCONDITION
        ):
            return cls._is_negated(a.content, b.content)

        return False

    @classmethod
    def _is_negated(cls, content_a: str, content_b: str) -> bool:
        """
        Heuristic check for logical negation between two content strings.

        Uses keyword pair matching to detect likely contradictions.
        """
        a_lower = content_a.strip().lower()
        b_lower = content_b.strip().lower()

        # Exact negation with "not"
        if a_lower == f"not {b_lower}" or b_lower == f"not {a_lower}":
            return True

        # Check negation keyword pairs
        for pos, neg in cls._NEGATION_PAIRS:
            a_has_pos = pos in a_lower
            a_has_neg = neg in a_lower
            b_has_pos = pos in b_lower
            b_has_neg = neg in b_lower

            # One has positive keyword, other has negative keyword
            if (a_has_pos and b_has_neg and not a_has_neg and not b_has_pos) or (
                a_has_neg and b_has_pos and not a_has_pos and not b_has_neg
            ):
                return True

        return False
