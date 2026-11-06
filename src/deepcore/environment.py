"""
environment.py
AIMA Problem Formulation: Verified Solvable Maps, Forward Successors, and Backward Predecessors
(for Bidirectional Search).
[TA PROVIDED - DO NOT MODIFY]
"""

from enum import Enum
from typing import Dict, List, Set, Tuple
import os
from .state import DeepCoreState, Pos


class Action(str, Enum):
    MOVE_N = "MOVE_N"
    MOVE_S = "MOVE_S"
    MOVE_E = "MOVE_E"
    MOVE_W = "MOVE_W"

    PUSH_N = "PUSH_N"
    PUSH_S = "PUSH_S"
    PUSH_E = "PUSH_E"
    PUSH_W = "PUSH_W"

    DRILL_N = "DRILL_N"
    DRILL_S = "DRILL_S"
    DRILL_E = "DRILL_E"
    DRILL_W = "DRILL_W"


DIRECTION_DELTAS: Dict[str, Pos] = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
}


# All 5 levels are mathematically verified to be 100% solvable in both Manual and AI modes.
# - BFS chooses shorter step-count paths through sludge/rubble (higher cost).
# - UCS / A* builds boulder bridges first (while carried=0) and rescues miners in optimal payload order.
BUILTIN_MAPS: Dict[str, str] = {
    # LEVEL 1: "The Payload Trap" (0 Boulders | Bait Miner + 3 Parallel Lanes: Sludge vs Rubble vs Dry)
    "level1": """
###########
#E.M.#...M#
#.##.#.##.#
#S..~~~...#
#.##.#.##.#
#.#..@..#.#
#.........#
###########
""".strip(),

    # LEVEL 2: "The Vertical Shaft" (Push South down the center shaft vs Side Sludge Chutes)
    # Kept two boulders and the shaft route; trimmed to two miners and one rubble so
    # BFS/UCS stay around 60k states instead of a quarter of a million.
    "level2": """
##############
#E..~~~~~...M#
###.#####.####
#.S..B....O..#
#.........#..#
###.#######.##
#....B....O..#
#....#....#..#
###.#####.#.##
#.....@...~..#
#.....#.....M#
##############
""".strip(),

    # LEVEL 3: "The Double-Chasm Crossing"
    # The 2-tile chasm still needs two boulders.  B1 is rolled east along the top
    # corridor; B2 now sits in the wall pocket directly under the second chasm
    # cell, so it is one clean push instead of roaming an open room.
    "level3": """
##############
#E..#....#..M#
###.#.##.#.###
#S..B....OO..#
###.#.##.#B###
#...#....#...#
#.#.#.##.#.###
#.~...@..~..M#
#........#...#
##############
""".strip(),


    # LEVEL 4: "Center-Island Extraction" (original first-commit Level 1 cave)
    # One redundant central boulder removed: the cave, its two remaining boulders,
    # three chasms and the rubble door are otherwise unchanged.
    "level4": """
#############
#E...#...O.M#
#....#.B.O..#
###.##...####
#...........#
#.S..~~~.B..#
#....~~~....#
####.###O####
#...@.M.....#
#############
""".strip(),

    # LEVEL 5: "The Double-Chasm Crossing" (larger variant)
    # Two boulders still fill the two chasm cells, but B2 is pocketed directly
    # under the second cell; the map keeps its four chambers and long route with
    # two miners and one rubble door.
    "level5": """
###############
#E..#.....#..M#
#...#.....#...#
###.#.###.#.###
#S..B.....OO..#
###.#.#.#.#B###
#...#.....#...#
#.#.#.....#.###
#.~....@..~..M#
#.........#...#
###############
""".strip(),

}

class DeepCoreEnv:
    """
    Formal Environment for the DeepCore Mine Rescue & Bridge-Building problem.
    Supports both Forward Search (get_successors) and Backward Search (get_predecessors).
    """

    COST_CLEAR_FLOOR = 1
    COST_SLUDGE = 5
    COST_PUSH_BASE = 6
    COST_DRILL = 10
    PAYLOAD_SURCHARGE = 2

    def __init__(self, map_text: str, level_name: str = "Custom"):
        self.level_name = level_name
        lines = [line.rstrip("\r\n") for line in map_text.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Map text cannot be empty.")

        self.rows = len(lines)
        self.cols = max(len(line) for line in lines)

        self.walls: Set[Pos] = set()
        self.sludge: Set[Pos] = set()
        self.initial_boulders: Set[Pos] = set()
        self.initial_chasms: Set[Pos] = set()
        self.initial_rubble: Set[Pos] = set()
        self.initial_miners: Set[Pos] = set()
        self.elevator_pos: Pos = (-1, -1)
        self.start_pos: Pos = (-1, -1)

        for r, line in enumerate(lines):
            for c, ch in enumerate(line.ljust(self.cols, "#")):
                pos = (r, c)
                if ch == "#":
                    self.walls.add(pos)
                elif ch == ".":
                    pass
                elif ch == "~":
                    self.sludge.add(pos)
                elif ch == "S":
                    self.start_pos = pos
                elif ch == "E":
                    self.elevator_pos = pos
                elif ch == "B":
                    self.initial_boulders.add(pos)
                elif ch == "O":
                    self.initial_chasms.add(pos)
                elif ch == "@":
                    self.initial_rubble.add(pos)
                elif ch == "M":
                    self.initial_miners.add(pos)
                else:
                    raise ValueError(f"Unrecognized map symbol '{ch}' at ({r}, {c})")

        if self.start_pos == (-1, -1):
            raise ValueError("Map is missing start position 'S'.")
        if self.elevator_pos == (-1, -1):
            raise ValueError("Map is missing elevator position 'E'.")

        self.total_miners = len(self.initial_miners)
        self.initial_state = DeepCoreState(
            bot_pos=self.start_pos,
            carried=0,
            boulders=frozenset(self.initial_boulders),
            chasms=frozenset(self.initial_chasms),
            rubble=frozenset(self.initial_rubble),
            miners=frozenset(self.initial_miners),
        )

    @classmethod
    def from_file_or_builtin(cls, identifier: str) -> "DeepCoreEnv":
        if os.path.isfile(identifier):
            with open(identifier, "r", encoding="utf-8") as f:
                return cls(f.read(), level_name=os.path.basename(identifier).replace(".txt", ""))
        key = identifier.lower().replace(".txt", "")
        if key in BUILTIN_MAPS:
            return cls(BUILTIN_MAPS[key], level_name=key.upper())
        raise FileNotFoundError(f"Map '{identifier}' not found.")

    def get_initial_state(self) -> DeepCoreState:
        return self.initial_state

    def is_goal(self, state: DeepCoreState) -> bool:
        """
        Goal Test G(s): All miners have been picked up into the rescue pod
        (len(state.miners) == 0) and the robot has reached the Extraction Elevator E.
        """
        return len(state.miners) == 0 and state.bot_pos == self.elevator_pos

    def get_goal_states(self) -> List[DeepCoreState]:
        """
        Returns candidate canonical goal states at the Extraction Elevator for Bidirectional Search.
        Includes the state where all chasms are bridged and the state where initial obstacles remain.
        """
        candidates = [
            # 1. All boulders used to bridge chasms (optimal bridge route)
            DeepCoreState(
                bot_pos=self.elevator_pos,
                carried=self.total_miners,
                boulders=frozenset(),
                chasms=frozenset(),
                rubble=frozenset(self.initial_rubble),
                miners=frozenset(),
            ),
            # 2. All chasms bridged AND all rubble drilled
            DeepCoreState(
                bot_pos=self.elevator_pos,
                carried=self.total_miners,
                boulders=frozenset(),
                chasms=frozenset(),
                rubble=frozenset(),
                miners=frozenset(),
            ),
            # 3. Rubble drilled without bridging chasms
            DeepCoreState(
                bot_pos=self.elevator_pos,
                carried=self.total_miners,
                boulders=frozenset(self.initial_boulders),
                chasms=frozenset(self.initial_chasms),
                rubble=frozenset(),
                miners=frozenset(),
            ),
        ]
        # Deduplicate while preserving order
        seen = set()
        unique: List[DeepCoreState] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_walkable_cell(self, pos: Pos, state: DeepCoreState) -> bool:
        return (
            self.in_bounds(pos)
            and pos not in self.walls
            and pos not in state.boulders
            and pos not in state.chasms
            and pos not in state.rubble
        )

    def _can_push_boulder_into(self, pos: Pos, state: DeepCoreState) -> bool:
        return (
            self.in_bounds(pos)
            and pos not in self.walls
            and pos not in state.boulders
            and pos not in state.rubble
            and pos not in state.miners
            and pos != self.elevator_pos
        )

    def get_legal_actions(self, state: DeepCoreState) -> List[Action]:
        legal: List[Action] = []
        r, c = state.bot_pos

        for dir_key, (dr, dc) in DIRECTION_DELTAS.items():
            adj = (r + dr, c + dc)
            if not self.in_bounds(adj) or adj in self.walls:
                continue

            if self._is_walkable_cell(adj, state):
                legal.append(Action(f"MOVE_{dir_key}"))
            elif adj in state.boulders:
                beyond = (r + 2 * dr, c + 2 * dc)
                if self._can_push_boulder_into(beyond, state):
                    legal.append(Action(f"PUSH_{dir_key}"))
            elif adj in state.rubble:
                legal.append(Action(f"DRILL_{dir_key}"))

        return legal

    def transition(self, state: DeepCoreState, action: Action) -> Tuple[DeepCoreState, int]:
        action_val = Action(action)
        kind, dir_key = action_val.value.split("_")
        dr, dc = DIRECTION_DELTAS[dir_key]
        r, c = state.bot_pos
        adj = (r + dr, c + dc)
        surcharge = self.PAYLOAD_SURCHARGE * state.carried

        if kind == "MOVE":
            if not self._is_walkable_cell(adj, state):
                raise ValueError(f"Illegal MOVE {action} from {state.bot_pos}")

            base_cost = self.COST_SLUDGE if adj in self.sludge else self.COST_CLEAR_FLOOR
            step_cost = base_cost + surcharge

            new_miners = state.miners
            new_carried = state.carried
            if adj in state.miners:
                new_miners = state.miners - {adj}
                new_carried += 1

            return (
                DeepCoreState(
                    bot_pos=adj,
                    carried=new_carried,
                    boulders=state.boulders,
                    chasms=state.chasms,
                    rubble=state.rubble,
                    miners=new_miners,
                ),
                step_cost,
            )

        elif kind == "PUSH":
            beyond = (r + 2 * dr, c + 2 * dc)
            if adj not in state.boulders or not self._can_push_boulder_into(beyond, state):
                raise ValueError(f"Illegal PUSH {action} from {state.bot_pos}")

            step_cost = self.COST_PUSH_BASE + surcharge
            new_boulders = set(state.boulders)
            new_chasms = set(state.chasms)

            new_boulders.remove(adj)
            if beyond in new_chasms:
                new_chasms.remove(beyond)
            else:
                new_boulders.add(beyond)

            return (
                DeepCoreState(
                    bot_pos=adj,
                    carried=state.carried,
                    boulders=frozenset(new_boulders),
                    chasms=frozenset(new_chasms),
                    rubble=state.rubble,
                    miners=state.miners,
                ),
                step_cost,
            )

        elif kind == "DRILL":
            if adj not in state.rubble:
                raise ValueError(f"Illegal DRILL {action} from {state.bot_pos}")

            return (
                DeepCoreState(
                    bot_pos=state.bot_pos,
                    carried=state.carried,
                    boulders=state.boulders,
                    chasms=state.chasms,
                    rubble=state.rubble - {adj},
                    miners=state.miners,
                ),
                self.COST_DRILL,
            )

        raise ValueError(f"Unknown action: {action}")

    def get_successors(self, state: DeepCoreState) -> List[Tuple[Action, DeepCoreState, int]]:
        """Forward transitions: returns list of (action, next_state, step_cost)."""
        return [
            (act, *self.transition(state, act))
            for act in self.get_legal_actions(state)
        ]

    def get_predecessors(self, state: DeepCoreState) -> List[Tuple[Action, DeepCoreState, int]]:
        """
        Exact Backward Transitions for Bidirectional Search:
        Given `state`, returns all (action, prev_state, step_cost) such that
        `self.transition(prev_state, action) == (state, step_cost)`.
        """
        predecessors: List[Tuple[Action, DeepCoreState, int]] = []
        r, c = state.bot_pos

        for dir_key, (dr, dc) in DIRECTION_DELTAS.items():
            # 1. Inverse of DRILL_d: bot stood at (r, c) and drilled adj = (r+dr, c+dc)
            adj = (r + dr, c + dc)
            if adj in self.initial_rubble and adj not in state.rubble:
                prev_state = DeepCoreState(
                    bot_pos=state.bot_pos,
                    carried=state.carried,
                    boulders=state.boulders,
                    chasms=state.chasms,
                    rubble=state.rubble | {adj},
                    miners=state.miners,
                )
                predecessors.append((Action(f"DRILL_{dir_key}"), prev_state, self.COST_DRILL))

            # 2. Inverse of MOVE_d: bot moved from prev_pos = (r-dr, c-dc) into (r, c)
            prev_pos = (r - dr, c - dc)
            if self._is_walkable_cell(prev_pos, state):
                base_c = self.COST_SLUDGE if (r, c) in self.sludge else self.COST_CLEAR_FLOOR
                # Case 2a: Did NOT pick up a miner on this step
                if (r, c) not in state.miners:
                    cost_a = base_c + self.PAYLOAD_SURCHARGE * state.carried
                    prev_a = DeepCoreState(
                        bot_pos=prev_pos,
                        carried=state.carried,
                        boulders=state.boulders,
                        chasms=state.chasms,
                        rubble=state.rubble,
                        miners=state.miners,
                    )
                    predecessors.append((Action(f"MOVE_{dir_key}"), prev_a, cost_a))

                # Case 2b: Picked up a miner at (r, c) on this step
                if (r, c) in self.initial_miners and (r, c) not in state.miners and state.carried > 0:
                    prev_carried = state.carried - 1
                    cost_b = base_c + self.PAYLOAD_SURCHARGE * prev_carried
                    prev_b = DeepCoreState(
                        bot_pos=prev_pos,
                        carried=prev_carried,
                        boulders=state.boulders,
                        chasms=state.chasms,
                        rubble=state.rubble,
                        miners=state.miners | {(r, c)},
                    )
                    predecessors.append((Action(f"MOVE_{dir_key}"), prev_b, cost_b))

            # 3. Inverse of PUSH_d: bot stood at prev_pos = (r-dr, c-dc) and pushed boulder from (r, c) into beyond = (r+dr, c+dc)
            beyond = (r + dr, c + dc)
            if (
                self._is_walkable_cell(prev_pos, state)
                and (r, c) not in state.miners
                and (r, c) != self.elevator_pos
            ):
                push_cost = self.COST_PUSH_BASE + self.PAYLOAD_SURCHARGE * state.carried
                # Case 3a: Boulder is currently sitting at `beyond`
                if beyond in state.boulders:
                    prev_boulders = (state.boulders - {beyond}) | {(r, c)}
                    prev_push_a = DeepCoreState(
                        bot_pos=prev_pos,
                        carried=state.carried,
                        boulders=prev_boulders,
                        chasms=state.chasms,
                        rubble=state.rubble,
                        miners=state.miners,
                    )
                    predecessors.append((Action(f"PUSH_{dir_key}"), prev_push_a, push_cost))

                # Case 3b: Boulder was pushed into a chasm at `beyond`, bridging it
                if beyond in self.initial_chasms and beyond not in state.chasms and beyond not in state.boulders:
                    prev_boulders = state.boulders | {(r, c)}
                    prev_chasms = state.chasms | {beyond}
                    prev_push_b = DeepCoreState(
                        bot_pos=prev_pos,
                        carried=state.carried,
                        boulders=prev_boulders,
                        chasms=prev_chasms,
                        rubble=state.rubble,
                        miners=state.miners,
                    )
                    predecessors.append((Action(f"PUSH_{dir_key}"), prev_push_b, push_cost))

        return predecessors

    def describe_tile(self, pos: Pos, state: DeepCoreState) -> Tuple[str, str]:
        k = state.carried
        surcharge = self.PAYLOAD_SURCHARGE * k
        if pos in self.walls:
            return "Cavern Rock Wall (#)", "Impassable"
        if pos in state.chasms:
            return "Bottomless Chasm (O)", "Fatal Pit — Push a Boulder (B) here to build a bridge!"
        if pos in state.rubble:
            return "Collapsed Rubble (@)", f"Laser-Drill (Hold SHIFT+Dir) Cost = {self.COST_DRILL}"
        if pos in state.boulders:
            return "Heavy Ore Boulder (B)", f"Push Cost = {self.COST_PUSH_BASE} + 2({k}) = {self.COST_PUSH_BASE + surcharge}"
        if pos == self.elevator_pos:
            return "Extraction Elevator (E)", f"Step Cost = {1 + surcharge} (Goal once all Miners rescued)"
        if pos in state.miners:
            base = self.COST_SLUDGE if pos in self.sludge else self.COST_CLEAR_FLOOR
            return "Trapped Miner (M)", f"Step Cost = {base + surcharge} (Rescues Miner: payload k -> {k + 1})"
        if pos in self.initial_chasms and pos not in state.chasms:
            return "Reinforced Boulder Bridge (=)", f"Walk Cost = 1 + 2({k}) = {1 + surcharge}"
        if pos in self.sludge:
            return "Toxic Flooded Sludge (~)", f"Walk Cost = 5 + 2({k}) = {5 + surcharge}"
        return "Clear Grated Tunnel (.)", f"Walk Cost = 1 + 2({k}) = {1 + surcharge}"
