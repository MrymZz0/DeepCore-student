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
@_timed
def a_star_search(
    env: DeepCoreEnv, heuristic: Callable[[DeepCoreState, DeepCoreEnv], float]
) -> Optional[SearchMetrics]:
    """
    A* Graph Search with f(n) = g(n) + h(n). Optimal when h is admissible.
    Heap entries are (f, h, counter, g, state): ties on f are broken by the smaller h
    (deeper nodes first). A state that is reached again with a smaller g is re-opened, so the
    result stays optimal even if h were not consistent. Children with h = inf are pruned.
    """
    metrics = SearchMetrics(algorithm_name="A*")
    start = env.get_initial_state()
    counter = itertools.count()

    h_cache: Dict[DeepCoreState, float] = {}

    def h_of(s: DeepCoreState) -> float:
        v = h_cache.get(s)
        if v is None:
            v = heuristic(s, env)
            h_cache[s] = v
        return v

    start_h = h_of(start)
    if start_h == INF:
        metrics.status_message = "Start state judged unsolvable by the heuristic."
        return metrics

    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    closed: Set[DeepCoreState] = set()
    heap: List[Tuple[float, float, int, int, DeepCoreState]] = [(start_h, start_h, next(counter), 0, start)]

    while heap:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(best_g) - len(closed))
        metrics.max_memory_size = max(metrics.max_memory_size, len(best_g))

        _, _, _, g, state = heapq.heappop(heap)
        if state in closed or g > best_g[state]:
            continue  # stale heap entry

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
            if old_g is not None:
                if new_g >= old_g:
                    continue
                closed.discard(child)  # cheaper route found: re-open
            h = h_of(child)
            if h == INF:
                continue
            best_g[child] = new_g
            parent[child] = (state, action.value)
            heapq.heappush(heap, (new_g + h, h, next(counter), new_g, child))

    metrics.status_message = "No solution found."
    return metrics


# =============================================================================
# BONUS 1: Weighted A* Search — [STUDENT BONUS TODO]
# =============================================================================
@_timed
def weighted_a_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    weight: float = 1.5,
) -> Optional[SearchMetrics]:
    """
    Weighted A*: f(n) = g(n) + W*h(n) with W >= 1. The returned cost is at most W times the
    optimal cost (for an admissible h); larger W usually expands fewer nodes. Same structure
    as A* (re-opening of states reached more cheaply, pruning of h = inf children).
    """
    if weight < 1.0:
        weight = 1.0
    metrics = SearchMetrics(algorithm_name=f"WA*(W={weight:g})")
    start = env.get_initial_state()
    counter = itertools.count()

    h_cache: Dict[DeepCoreState, float] = {}

    def h_of(s: DeepCoreState) -> float:
        v = h_cache.get(s)
        if v is None:
            v = heuristic(s, env)
            h_cache[s] = v
        return v

    start_h = h_of(start)
    if start_h == INF:
        metrics.status_message = "Start state judged unsolvable by the heuristic."
        return metrics

    best_g: Dict[DeepCoreState, int] = {start: 0}
    parent: Dict[DeepCoreState, Tuple[Optional[DeepCoreState], Optional[str]]] = {start: (None, None)}
    closed: Set[DeepCoreState] = set()
    heap: List[Tuple[float, float, int, int, DeepCoreState]] = [
        (weight * start_h, start_h, next(counter), 0, start)
    ]

    while heap:
        metrics.max_frontier_size = max(metrics.max_frontier_size, len(best_g) - len(closed))
        metrics.max_memory_size = max(metrics.max_memory_size, len(best_g))

        _, _, _, g, state = heapq.heappop(heap)
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
            if old_g is not None:
                if new_g >= old_g:
                    continue
                closed.discard(child)
            h = h_of(child)
            if h == INF:
                continue
            best_g[child] = new_g
            parent[child] = (state, action.value)
            heapq.heappush(heap, (new_g + weight * h, h, next(counter), new_g, child))

    metrics.status_message = "No solution found."
    return metrics




# =============================================================================
# BONUS 2: Iterative Deepening A* (IDA*) — [STUDENT BONUS TODO]
# =============================================================================
@_timed
def ida_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    max_expansions: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    IDA*: repeated depth-first contours bounded by f = g + h <= threshold; the next
    threshold is the smallest f that exceeded the current one.

    Because the same state is reachable by very many commuting move orders, each
    contour keeps a transposition table {state: lowest g seen in this contour}; a state is
    skipped if it is reached again with g >= the stored value. The table is thrown away
    after every contour, so memory is bounded by one contour (still far smaller than
    A*'s closed list on the early contours).
    """
    metrics = SearchMetrics(algorithm_name="IDA*")
    start = env.get_initial_state()
    h_cache: Dict[DeepCoreState, float] = {}

    def h_of(s: DeepCoreState) -> float:
        v = h_cache.get(s)
        if v is None:
            v = heuristic(s, env)
            h_cache[s] = v
        return v

    threshold = h_of(start)
    if threshold == INF:
        metrics.status_message = "Start state judged unsolvable by the heuristic."
        return metrics

    while True:
        table: Dict[DeepCoreState, int] = {}
        next_threshold = INF
        stack: List[Tuple[DeepCoreState, int, "_Node"]] = [(start, 0, None)]

        while stack:
            metrics.max_frontier_size = max(metrics.max_frontier_size, len(stack))
            metrics.max_memory_size = max(metrics.max_memory_size, len(stack) + len(table))
            state, g, node = stack.pop()

            known = table.get(state)
            if known is not None and known <= g:
                continue
            f = g + h_of(state)
            if f > threshold:
                if f < next_threshold:
                    next_threshold = f
                continue
            table[state] = g

            if env.is_goal(state):
                metrics.solved = True
                metrics.solution_actions = _actions_from_node(node)
                metrics.total_cost = g
                metrics.status_message = f"Solved within f-threshold {threshold:g}."
                return metrics

            if max_expansions is not None and metrics.nodes_expanded >= max_expansions:
                metrics.status_message = f"IDA* stopped: expansion budget ({max_expansions}) exhausted."
                return metrics

            metrics.nodes_expanded += 1
            metrics.explored_order.append(state.bot_pos)

            children = []
            for action, child, step_cost in env.get_successors(state):
                metrics.nodes_generated += 1
                new_g = g + step_cost
                k = table.get(child)
                if k is not None and k <= new_g:
                    continue
                hc = h_of(child)
                if hc == INF:
                    continue
                children.append((new_g + hc, hc, action.value, child, new_g))
            # push worst-first so that the most promising child is popped first
            children.sort(key=lambda t: (t[0], t[1]), reverse=True)
            for _, _, act_name, child, new_g in children:
                stack.append((child, new_g, (act_name, node)))

        if next_threshold == INF:
            metrics.status_message = "No solution found (all contours exhausted)."
            return metrics
        threshold = next_threshold


# =============================================================================
# BONUS 3: Simplified Memory-Bounded A* (SMA*) — [STUDENT BONUS TODO]
# =============================================================================
class _SMANode:
    """One node of the SMA* search tree (a *tree*, so a state may appear several times)."""

    __slots__ = ("state", "parent", "action", "g", "f", "depth", "children", "forgotten", "alive", "ver", "uid")

    def __init__(self, state, parent, action, g, f, depth, uid):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.f = f                  # backed-up f value
        self.depth = depth
        self.children: Dict[str, "_SMANode"] = {}   # successors currently in memory
        self.forgotten: Dict[str, float] = {}       # action -> f of a successor that was deleted
        self.alive = True
        self.ver = 0                # lazy-deletion version for the two heaps
        self.uid = uid


_SMA_DEFAULT_MEMORY = 20000
_SMA_MAX_EXPANSIONS = 400_000     # safety net so a thrashing run (memory far too small) always terminates


@_timed
def sma_star_search(
    env: DeepCoreEnv,
    heuristic: Callable[[DeepCoreState, DeepCoreEnv], float],
    memory_limit: Optional[int] = None,
) -> Optional[SearchMetrics]:
    """
    SMA*: A* that never keeps more than `memory_limit` search-tree nodes.

    * Always expand the best leaf (lowest f, deepest on ties). All its successors are added;
      each child gets f = max(f(parent), g + h) (path-max).
    * When memory is full, delete the *worst* leaf (highest f, shallowest on ties). Its f is
      remembered in the parent's `forgotten` table, so the knowledge is not lost.
    * Backed-up values: f(node) = min over its children (in memory or forgotten); changes are
      propagated to the ancestors. When all children of a node are deleted it becomes a leaf
      again with its backed-up f, and is regenerated later if that f becomes the best.
    * A leaf whose f is infinity (dead end / no room) can never lead to a solution.
    Complete only if the shallowest solution fits into memory; optimal if it fits and h is
    admissible. A successor whose state is already in memory with a smaller or equal g is
    dominated and dropped (this also prevents cycles).
    """
    metrics = SearchMetrics(algorithm_name="SMA*")
    limit = _SMA_DEFAULT_MEMORY if memory_limit is None else int(memory_limit)
    if limit < 2:
        metrics.status_message = "memory_limit must be at least 2 nodes."
        return metrics

    start = env.get_initial_state()
    h_cache: Dict[DeepCoreState, float] = {}

    def h_of(s: DeepCoreState) -> float:
        v = h_cache.get(s)
        if v is None:
            v = heuristic(s, env)
            h_cache[s] = v
        return v

    h0 = h_of(start)
    if h0 == INF:
        metrics.status_message = "Start state judged unsolvable by the heuristic."
        return metrics

    uid_counter = itertools.count()
    root = _SMANode(start, None, None, 0, h0, 0, next(uid_counter))
    best_heap: List[tuple] = []    # (f, -depth, uid, ver, node)   -> best leaf
    worst_heap: List[tuple] = []   # (-f, depth, uid, ver, node)   -> worst leaf
    state_count = {"memory": 1, "leaves": 0}

    def push_leaf(node: _SMANode) -> None:
        node.ver += 1
        heapq.heappush(best_heap, (node.f, -node.depth, node.uid, node.ver, node))
        heapq.heappush(worst_heap, (-node.f, node.depth, node.uid, node.ver, node))
        state_count["leaves"] += 1

    def is_valid_leaf(node: _SMANode, ver: int) -> bool:
        return node.alive and not node.children and node.ver == ver

    def pop_best() -> Optional[_SMANode]:
        while best_heap:
            _, _, _, ver, node = heapq.heappop(best_heap)
            if is_valid_leaf(node, ver):
                return node
        return None

    def delete_worst() -> bool:
        """Forget the worst leaf. Returns False if nothing can be deleted."""
        while worst_heap:
            _, _, _, ver, node = heapq.heappop(worst_heap)
            if not is_valid_leaf(node, ver):
                continue
            parent = node.parent
            if parent is None:
                return False
            node.alive = False
            state_count["memory"] -= 1
            state_count["leaves"] -= 1
            if in_memory.get(node.state) is node:
                del in_memory[node.state]
            del parent.children[node.action]
            parent.forgotten[node.action] = node.f
            if not parent.children:  # parent turns into a leaf with its backed-up value
                parent.f = min(parent.forgotten.values())
                push_leaf(parent)
            return True
        return False

    def backup(node: Optional[_SMANode]) -> None:
        while node is not None:
            values = [c.f for c in node.children.values()]
            values.extend(node.forgotten.values())
            if not values:
                return
            new_f = min(values)
            if new_f == node.f:
                return
            node.f = new_f
            node = node.parent

    # Duplicate detection inside the memory: state -> cheapest node currently held for it.
    # A successor whose state is already in memory with g <= its own g is dominated and is
    # not stored (this also removes cycles, since ancestors have smaller g).
    in_memory: Dict[DeepCoreState, _SMANode] = {start: root}

    def dominated(state: DeepCoreState, g: int) -> bool:
        other = in_memory.get(state)
        return other is not None and other.g <= g

    push_leaf(root)

    while True:
        metrics.max_frontier_size = max(metrics.max_frontier_size, state_count["leaves"])
        metrics.max_memory_size = max(metrics.max_memory_size, state_count["memory"])

        node = pop_best()
        if node is None or node.f == INF:
            metrics.status_message = (
                f"SMA* failed: no solution fits in {limit} nodes (or the space is exhausted)."
            )
            return metrics
        state_count["leaves"] -= 1

        if env.is_goal(node.state):
            actions: List[str] = []
            cur = node
            while cur.parent is not None:
                actions.append(cur.action)
                cur = cur.parent
            actions.reverse()
            metrics.solved = True
            metrics.solution_actions = actions
            metrics.total_cost = node.g
            metrics.status_message = f"Solved with memory_limit={limit}."
            return metrics

        if metrics.nodes_expanded >= _SMA_MAX_EXPANSIONS:
            metrics.status_message = "SMA* stopped: expansion safety limit reached."
            return metrics
        metrics.nodes_expanded += 1
        metrics.explored_order.append(node.state.bot_pos)

        # ---- successors: first expansion, or regeneration of forgotten ones ----
        specs: List[Tuple[str, DeepCoreState, int, float]] = []   # (action, state, step_cost, f)
        if node.forgotten:
            regenerate = node.forgotten
            node.forgotten = {}
            for act_name, f_val in regenerate.items():
                child_state, step_cost = env.transition(node.state, act_name)
                specs.append((act_name, child_state, step_cost, f_val))
        else:
            for action, child_state, step_cost in env.get_successors(node.state):
                h = h_of(child_state)
                if h == INF:
                    continue
                f_val = max(node.f, node.g + step_cost + h)   # path-max
                specs.append((action.value, child_state, step_cost, f_val))
        metrics.nodes_generated += len(specs)

        added = 0
        for act_name, child_state, step_cost, f_val in specs:
            child_g = node.g + step_cost
            if dominated(child_state, child_g):
                continue
            if node.depth + 1 >= limit and not env.is_goal(child_state):
                f_val = INF   # a path this long cannot be held in memory, so it cannot be a solution
            child = _SMANode(child_state, node, act_name, child_g, f_val,
                             node.depth + 1, next(uid_counter))
            node.children[act_name] = child
            old = in_memory.get(child_state)
            if old is None or child_g < old.g:
                in_memory[child_state] = child
            state_count["memory"] += 1
            added += 1
            push_leaf(child)
            while state_count["memory"] > limit:
                if not delete_worst():
                    metrics.status_message = f"SMA* failed: memory_limit={limit} is too small."
                    return metrics
            metrics.max_memory_size = max(metrics.max_memory_size, state_count["memory"])

        if added == 0:          # dead end (or everything dominated): can never lead to a goal
            node.f = INF
            push_leaf(node)
            backup(node.parent)
            continue
        if not node.children:
            # Every child was pushed out again. delete_worst() already turned `node` back into a
            # leaf with its backed-up f; it is simply retried when it is the best leaf again.
            continue
        backup(node)

