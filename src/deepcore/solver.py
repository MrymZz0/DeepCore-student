"""
solver.py
Student implementation file for DeepCore search algorithms.

- Breadth-First Search (BFS) is provided below as a complete reference example
  showing how to interact with `env.get_successors(state)` and `SearchMetrics`.
- Implement the remaining required algorithms. Do not change the public
  function signatures or the environment/state modules.
"""
import itertools
from collections import deque
import heapq
import math
from typing import Callable, Dict, List, Optional, Set, Tuple
import functools
import time
from .environment import DeepCoreEnv
from .state import DeepCoreState, SearchMetrics


def _reconstruct_path(
    parent_map: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]],
    goal_state: DeepCoreState,
) -> List[str]:
    """Helper function to trace actions from the initial state to `goal_state`."""
    actions: List[str] = []
    curr: Optional[DeepCoreState] = goal_state
    while curr is not None:
        prev, act = parent_map.get(curr, (None, None))
        if act is not None:
            actions.append(act)
        curr = prev
    actions.reverse()
    return actions


# =============================================================================
# 1. Breadth-First Search (BFS) — [PROVIDED AS REFERENCE EXAMPLE]
# =============================================================================
def breadth_first_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:
    """
    Reference Graph-Search BFS implementation (FIFO queue).
    Minimizes the number of steps, but ignores terrain/payload energy costs!
    """
    metrics = SearchMetrics(algorithm_name="BFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        return metrics

    frontier = deque([start])
    frontier_set = {start}
    explored: Set[DeepCoreState] = set()
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    cost_so_far: Dict[DeepCoreState, int] = {start: 0}

    while frontier:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(frontier))
        metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))
        state = frontier.popleft()
        frontier_set.discard(state)

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, next_state, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if next_state not in explored and next_state not in frontier_set:
                parent[next_state] = (state, action.value)
                cost_so_far[next_state] = cost_so_far[state] + step_cost
                if env.is_goal(next_state):
                    metrics.solved = True
                    metrics.solution_actions = _reconstruct_path(parent, next_state)
                    metrics.total_cost = cost_so_far[next_state]
                    return metrics
                frontier.append(next_state)
                frontier_set.add(next_state)
                metrics.max_memory_size = max(metrics.max_memory_size, len(frontier) + len(explored))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 2. Depth-First Search (DFS) & Iterative Deepening Search (IDS) — [STUDENT TODO]
# =============================================================================
INF = math.inf
_Node = Optional[Tuple[str, "_Node"]]


def _actions_from_node(node: "_Node") -> List[str]:
    actions: List[str] = []
    while node is not None:
        actions.append(node[0])
        node = node[1]
    actions.reverse()
    return actions


def _timed(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        metrics = fn(*args, **kwargs)
        if metrics is not None:
            metrics.runtime_sec = time.perf_counter() - t0
        return metrics
    return wrapper

@_timed
def depth_first_search(env: DeepCoreEnv, max_expansions: Optional[int] = None) -> Optional[SearchMetrics]:
    metrics = SearchMetrics(algorithm_name="DFS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Start state is already a goal."
        return metrics

    stack: List[Tuple[DeepCoreState, "_Node", int]] = [(start, None, 0)]
    explored: Set[DeepCoreState] = set()

    while stack:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
        metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(explored))
        state, node, g = stack.pop()
        if state in explored:
            continue

        if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
            metrics.status_message = f"DFS stopped: expansion budget ({max_expansions}) exhausted."
            return metrics

        explored.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in reversed(env.get_successors(state)):
            metrics.nodes_generated += 1
            if child in explored:
                continue
            child_node = (action.value, node)
            if env.is_goal(child):
                metrics.solved = True
                metrics.solution_actions = _actions_from_node(child_node)
                metrics.total_cost = g + step_cost
                return metrics
            stack.append((child, child_node, g + step_cost))

    metrics.status_message = "No solution found."
    return metrics


@_timed
def iterative_deepening_search(
    env: DeepCoreEnv,
    max_depth: Optional[int] = None,
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="IDS")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Start state is already a goal."
        return metrics

    limit = 0
    while max_depth is None or limit <= max_depth:
        best_depth: Dict[DeepCoreState, int] = {}
        stack: List[Tuple[DeepCoreState, int, "_Node", int]] = [(start, 0, None, 0)]
        cutoff = False

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(best_depth))
            state, depth, node, g = stack.pop()

            seen_depth = best_depth.get(state)
            if seen_depth is not None and seen_depth <= depth:
                continue
            best_depth[state] = depth

            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = f"IDS stopped: expansion budget ({max_expansions}) exhausted."
                return metrics

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _actions_from_node(node)
                metrics.total_cost = g
                metrics.status_message = f"Solved at depth limit {limit}."
                return metrics

            if depth == limit:
                if not cutoff and env.get_legal_actions(state):
                    cutoff = True
                continue

            for action, child, step_cost in reversed(env.get_successors(state)):
                metrics.nodes_generated += 1
                prev = best_depth.get(child)
                if prev is None or prev > depth + 1:
                    stack.append((child, depth + 1, (action.value, node), g + step_cost))

        if not cutoff:
            metrics.status_message = "No solution found (state space exhausted)."
            return metrics
        limit += 1

    metrics.status_message = f"No solution within max_depth={max_depth}."
    return metrics


# =============================================================================
# 3. Bidirectional Search — [STUDENT TODO]
# =============================================================================
def _backward_seed_states(env: DeepCoreEnv) -> List[DeepCoreState]:
    seeds: List[DeepCoreState] = list(env.get_goal_states())
    seen: Set[DeepCoreState] = set(seeds)
    chasms = sorted(env.initial_chasms)
    boulders = sorted(env.initial_boulders)
    rubble_options = [frozenset(env.initial_rubble), frozenset()]
    budget = 400

    for k in range(0, min(len(chasms), len(boulders)) + 1):
        for filled in itertools.combinations(chasms, k):
            for used in itertools.combinations(boulders, k):
                for rub in rubble_options:
                    if len(seeds) >= budget:
                        return seeds
                    cand = DeepCoreState(
                        bot_pos=env.elevator_pos,
                        carried=env.total_miners,
                        boulders=frozenset(env.initial_boulders) - frozenset(used),
                        chasms=frozenset(env.initial_chasms) - frozenset(filled),
                        rubble=rub,
                        miners=frozenset(),
                    )
                    if cand not in seen and env.is_goal(cand):
                        seen.add(cand)
                        seeds.append(cand)
    return seeds


@_timed
def bidirectional_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="Bidirectional")
    start = env.get_initial_state()
    if env.is_goal(start):
        metrics.solved = True
        metrics.status_message = "Start state is already a goal."
        return metrics

    Parent = Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]]
    fwd_parent: Parent = {start: (None, None)}
    fwd_g: Dict[DeepCoreState, int] = {start: 0}
    fwd_d: Dict[DeepCoreState, int] = {start: 0}
    bwd_parent: Parent = {}
    bwd_g: Dict[DeepCoreState, int] = {}
    bwd_d: Dict[DeepCoreState, int] = {}

    fwd_layer: List[DeepCoreState] = [start]
    bwd_layer: List[DeepCoreState] = []
    for goal in _backward_seed_states(env):
        if goal not in bwd_parent:
            bwd_parent[goal] = (None, None)
            bwd_g[goal] = 0
            bwd_d[goal] = 0
            bwd_layer.append(goal)

    def build(meeting: DeepCoreState) -> None:
        actions: List[str] = []
        cur = meeting
        while True:
            prev, act = fwd_parent[cur]
            if prev is None:
                break
            actions.append(act)
            cur = prev
        actions.reverse()
        cur = meeting
        while True:
            nxt, act = bwd_parent[cur]
            if nxt is None:
                break
            actions.append(act)
            cur = nxt
        metrics.solved = True
        metrics.solution_actions = actions
        metrics.total_cost = fwd_g[meeting] + bwd_g[meeting]

    if start in bwd_parent:
        build(start)
        return metrics

    while fwd_layer and bwd_layer:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(fwd_layer) + len(bwd_layer))
        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_parent) + len(bwd_parent))

        forward = len(fwd_layer) <= len(bwd_layer)
        layer = fwd_layer if forward else bwd_layer
        this_parent, this_g, this_d = (fwd_parent, fwd_g, fwd_d) if forward else (bwd_parent, bwd_g, bwd_d)
        other_parent = bwd_parent if forward else fwd_parent

        next_layer: List[DeepCoreState] = []
        best_meeting: Optional[DeepCoreState] = None
        best_key: Tuple[float, float] = (INF, INF)

        for state in layer:
            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)
            moves = env.get_successors(state) if forward else env.get_predecessors(state)
            for action, other, step_cost in moves:
                metrics.nodes_generated += 1
                if other in this_parent:
                    continue
                this_parent[other] = (state, action.value)
                this_g[other] = this_g[state] + step_cost
                this_d[other] = this_d[state] + 1
                next_layer.append(other)

                if forward and other not in other_parent and env.is_goal(other):
                    bwd_parent[other] = (None, None)
                    bwd_g[other] = 0
                    bwd_d[other] = 0
                if other in other_parent:
                    key = (fwd_d[other] + bwd_d[other], fwd_g[other] + bwd_g[other])
                    if key < best_key:
                        best_key = key
                        best_meeting = other

        metrics.max_memory_size = max(metrics.max_memory_size, len(fwd_parent) + len(bwd_parent))
        if best_meeting is not None:
            build(best_meeting)
            return metrics

        if forward:
            fwd_layer = next_layer
        else:
            bwd_layer = next_layer

    metrics.status_message = "No solution found (one search space exhausted)."
    return metrics


# =============================================================================
# 4. Uniform Cost Search (UCS) — [STUDENT TODO]
# =============================================================================
@_timed
def uniform_cost_search(env: DeepCoreEnv) -> Optional[SearchMetrics]:

    metrics = SearchMetrics(algorithm_name="UCS")
    start = env.get_initial_state()
    counter = itertools.count()

    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    closed: Set[DeepCoreState] = set()
    heap: List[Tuple[int, int, DeepCoreState]] = [(0, next(counter), start)]

    while heap:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(best_g) - len(closed))
        metrics.max_memory_size = max(metrics.max_memory_size, len(best_g))

        g, _, state = heapq.heappop(heap)
        if state in closed or g > best_g[state]:
            continue

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = g
            return metrics

        closed.add(state)
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            new_g = g + step_cost
            old_g = best_g.get(child)
            if old_g is not None and new_g >= old_g:
                continue
            best_g[child] = new_g
            parent[child] = (state, action.value)
            heapq.heappush(heap, (new_g, next(counter), child))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# 5. Greedy Best-First Search (GBFS) — [STUDENT TODO]
# =============================================================================
@_timed
def greedy_best_first_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """
    Greedy Best-First Search: order the frontier strictly by h(n), ignoring g(n).
    Fast, but neither optimal nor guaranteed to find a cheap plan. A state is accepted the
    first time it is discovered (no re-opening). Children with h = inf (proven dead ends)
    are pruned.
    """
    metrics = SearchMetrics(algorithm_name="GBFS")
    start = env.get_initial_state()
    counter = itertools.count()

    start_h = heuristic(start, env)
    if start_h == INF:
        metrics.status_message = "Start state judged unsolvable by the heuristic."
        return metrics

    # state -> g of the path through which it was discovered (also acts as the "seen" set)
    g_of: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    heap: List[Tuple[float, int, DeepCoreState]] = [(start_h, next(counter), start)]
    expanded = 0  # states already popped; frontier = discovered - expanded

    while heap:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(g_of) - expanded)
        metrics.max_memory_size = max(metrics.max_memory_size, len(g_of))

        _, _, state = heapq.heappop(heap)

        if env.is_goal(state):
            metrics.solved = True
            metrics.solution_actions = _reconstruct_path(parent, state)
            metrics.total_cost = g_of[state]
            return metrics

        expanded += 1
        metrics.nodes_expanded += 1
        metrics.explored_order.append(state.bot_pos)

        for action, child, step_cost in env.get_successors(state):
            metrics.nodes_generated += 1
            if child in g_of:
                continue
            h = heuristic(child, env)
            if h == INF:
                continue
            g_of[child] = g_of[state] + step_cost
            parent[child] = (state, action.value)
            heapq.heappush(heap, (h, next(counter), child))

    metrics.status_message = "No solution found."
    return metrics



# =============================================================================
# 6. A* Search — [STUDENT TODO]
# =============================================================================
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """TODO: Implement A* Graph Search ordering the priority queue by f(n) = g(n) + h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Weighted A* ordering the frontier by f(n) = g(n) + W * h(n)."""
    raise NotImplementedError


# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Iterative Deepening A* (IDA*) using f-cost threshold contours."""
    raise NotImplementedError


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """TODO (Bonus): Implement Simplified Memory-Bounded A* (SMA*)."""
    raise NotImplementedError
