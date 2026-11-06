"""
state.py
Immutable State Representation and Standardized Search Metrics for DeepCore.
[TA PROVIDED - DO NOT MODIFY]
"""

from dataclasses import dataclass, field
from typing import Tuple, FrozenSet, List

Pos = Tuple[int, int]


@dataclass(frozen=True)
class DeepCoreState:
    """
    Formal immutable and hashable state s for the DeepCore search problem.

    Attributes:
        bot_pos: (row, col) coordinates of the rescue robot.
        carried: Number of rescued miners currently inside the pressurized pod (+2 cost/step each).
        boulders: Frozenset of (row, col) coordinates of movable boulders.
        chasms: Frozenset of (row, col) coordinates of unbridged bottomless chasms.
        rubble: Frozenset of (row, col) coordinates of undrilled rubble piles.
        miners: Frozenset of (row, col) coordinates of waiting miners still on the map.
    """
    bot_pos: Pos
    carried: int
    boulders: FrozenSet[Pos]
    chasms: FrozenSet[Pos]
    rubble: FrozenSet[Pos]
    miners: FrozenSet[Pos]

    def __lt__(self, other: "DeepCoreState") -> bool:
        if not isinstance(other, DeepCoreState):
            return NotImplemented
        return (
            len(self.miners),
            -self.carried,
            len(self.chasms),
            len(self.rubble),
            self.bot_pos,
        ) < (
            len(other.miners),
            -other.carried,
            len(other.chasms),
            len(other.rubble),
            other.bot_pos,
        )


@dataclass
class SearchMetrics:
    """Standardized return object for all uninformed, informed, and bonus search algorithms."""
    algorithm_name: str = "Unknown"
    solved: bool = False
    solution_actions: List[str] = field(default_factory=list)
    total_cost: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    max_memory_size: int = 0
    runtime_sec: float = 0.0
    explored_order: List[Pos] = field(default_factory=list)
    status_message: str = "Ready"