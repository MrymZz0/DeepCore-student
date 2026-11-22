"""
heuristics.py
Student heuristic starter file for DeepCore:
  - h_naive_manhattan : provided baseline
  - h1_basic          : required student implementation
  - h2_advanced       : required student implementation
"""

import heapq
from typing import Dict, FrozenSet, List, Tuple
from collections import deque
from .environment import DeepCoreEnv, DIRECTION_DELTAS
from .state import DeepCoreState, Pos

# NOTE: caches are stored on the env object itself (not keyed by id(env)),
# because id() values can be reused after garbage collection, which would
# silently return distances of a *different* map.
_WALL_DIST_CACHE: Dict[int, Dict[Tuple[Pos, Pos], int]] = {}  # kept for compatibility, unused

INF = float("inf")


def _env_cache(env: DeepCoreEnv) -> dict:
    cache = getattr(env, "_deepcore_h_cache", None)
    if cache is None:
        cache = {}
        env._deepcore_h_cache = cache
    return cache


def _adjacency(env: DeepCoreEnv) -> Dict[Pos, List[Pos]]:
    """Neighbours of every non-wall cell (walls only; dynamic obstacles ignored)."""
    cache = _env_cache(env)
    adj = cache.get("adj")
    if adj is None:
        adj = {}
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.walls:
                    continue
                nbrs = []
                for dr, dc in DIRECTION_DELTAS.values():
                    nxt = (r + dr, c + dc)
                    if env.in_bounds(nxt) and nxt not in env.walls:
                        nbrs.append(nxt)
                adj[(r, c)] = nbrs
        cache["adj"] = adj
    return adj


def _get_wall_aware_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """All-pairs shortest path *step counts* ignoring dynamic obstacles (only walls # block)."""
    cache = _env_cache(env)
    dist_map = cache.get("steps")
    if dist_map is not None:
        return dist_map

    adj = _adjacency(env)
    dist_map = {}
    for src in adj:
        q = deque([src])
        dist_map[(src, src)] = 0
        seen = {src}
        while q:
            curr = q.popleft()
            d = dist_map[(src, curr)]
            for nxt in adj[curr]:
                if nxt not in seen:
                    seen.add(nxt)
                    dist_map[(src, nxt)] = d + 1
                    q.append(nxt)
    cache["steps"] = dist_map
    return dist_map


def _enter_cost(env: DeepCoreEnv, pos: Pos) -> int:
    """Cheapest possible base cost of a bot move that ends on `pos` (sludge = 5, else 1)."""
    return env.COST_SLUDGE if pos in env.sludge else env.COST_CLEAR_FLOOR


def _get_weighted_distances(env: DeepCoreEnv) -> Dict[Tuple[Pos, Pos], int]:
    """
    All-pairs *base-cost* distances ignoring dynamic obstacles and payload.
    Entering a sludge cell costs 5, any other cell 1. Every real action that moves
    the bot onto a cell costs at least this much (a PUSH costs 6 >= 5), so the
    result never exceeds the true base cost of travelling between two cells.
    """
    cache = _env_cache(env)
    wd = cache.get("wcost")
    if wd is not None:
        return wd

    adj = _adjacency(env)
    wd = {}
    for src in adj:
        best = {src: 0}
        heap = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > best[u]:
                continue
            wd[(src, u)] = d
            for v in adj[u]:
                nd = d + _enter_cost(env, v)
                if nd < best.get(v, INF):
                    best[v] = nd
                    heapq.heappush(heap, (nd, v))
    cache["wcost"] = wd
    return wd


def _prim_mst_cost(nodes: List[Pos], dist_fn) -> float:
    """Computes Minimum Spanning Tree weight over `nodes` using `dist_fn(u, v)`."""
    if len(nodes) <= 1:
        return 0.0
    in_mst = {nodes[0]}
    total_weight = 0.0

    while len(in_mst) < len(nodes):
        best_edge = float("inf")
        best_node = None
        for u in in_mst:
            for v in nodes:
                if v not in in_mst:
                    w = dist_fn(u, v)
                    if w < best_edge:
                        best_edge = w
                        best_node = v
        if best_node is None:
            break
        in_mst.add(best_node)
        total_weight += best_edge

    return total_weight


def h_naive_manhattan(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Baseline Naive Heuristic: Manhattan distance to nearest waiting miner (or Elevator if none).
    """
    if env.is_goal(state):
        return 0.0
    r, c = state.bot_pos
    if state.miners:
        return float(min(abs(r - mr) + abs(c - mc) for mr, mc in state.miners))
    er, ec = env.elevator_pos
    return float(abs(r - er) + abs(c - ec))


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h1_basic(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Admissible AND consistent MST + payload lower bound (Manhattan geometry).

        T  = {E} U miners                      (targets that must all be visited)
        h1 = MST_Manhattan(T) + min_{t in T} d(bot, t)
             + 2*k*d(bot, E) + 2 * sum_{m in miners} d(m, E)

    * The walk starts at the bot, first reaches some target (>= min distance) and then has to
      visit every target, which costs at least the MST weight of T (>= 1 per move/push).
      This is >= MST_Manhattan({bot} U T) of the textbook formula, so it dominates it.
      Putting the bot *inside* the MST is not consistent (the MST weight is not monotone in
      the vertex set), whereas "MST of the targets + distance to the nearest target" is.
    * Each of the k miners already aboard adds 2 per move/push for at least d(bot, E) more of
      them; each waiting miner adds 2 for at least d(m, E).
    * DRILL / extra PUSH cost / sludge are ignored (they only add cost).
    """
    if env.is_goal(state):
        return 0.0

    bot = state.bot_pos
    elev = env.elevator_pos
    targets = list({elev, *state.miners})
    mst = _prim_mst_cost(targets, _manhattan)
    nearest = min(_manhattan(bot, t) for t in targets)

    payload = 2 * state.carried * _manhattan(bot, elev)
    payload += 2 * sum(_manhattan(m, elev) for m in state.miners)
    return float(mst + nearest + payload)


def _is_corner_deadlock(pos: Pos, env: DeepCoreEnv) -> bool:
    """Returns True if a boulder at `pos` is wedged against perpendicular rock walls (#)."""
    r, c = pos
    n_wall = (r - 1, c) in env.walls
    s_wall = (r + 1, c) in env.walls
    e_wall = (r, c + 1) in env.walls
    w_wall = (r, c - 1) in env.walls
    return (n_wall or s_wall) and (e_wall or w_wall)


def _reach(adj: Dict[Pos, List[Pos]], src: Pos, blocked) -> set:
    """Cells reachable from `src` when walls and the `blocked` cells cannot be entered."""
    seen = {src}
    stack = [src]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen and v not in blocked:
                seen.add(v)
                stack.append(v)
    return seen


def _chasm_open_extra(env: DeepCoreEnv, pos: Pos) -> int:
    """
    Unavoidable *extra* cost of bridging chasm `pos`, on top of the one move that the
    distance term already pays for: a PUSH costs 6 and its bot move ends on the cell
    the boulder came from (a neighbour of `pos`), whose own entry cost is already
    counted. Take the minimum over all possible neighbours to stay admissible.
    """
    adj = _adjacency(env)
    options = [env.COST_PUSH_BASE - _enter_cost(env, q) for q in adj.get(pos, [])]
    return min(options) if options else 0


def _barrier_analysis(env, bot, chasms, rubble, miners, dead):
    """
    Returns (feasible, extra_cost, n_essential_chasms).

    * Deadlocked boulders (wedged in a wall corner) can never move again, so they act
      as permanent walls. If a target (miner / elevator) is then unreachable even when
      every chasm and rubble pile is treated as passable, the state is unsolvable.
    * A chasm / rubble cell is *essential* if blocking just that cell disconnects a
      target from the bot: every plan must then pay to open it (PUSH or DRILL).
    """
    adj = _adjacency(env)
    targets = set(miners)
    targets.add(env.elevator_pos)
    base_blocked = set(dead)

    if not targets <= _reach(adj, bot, base_blocked):
        return False, 0, 0

    extra = 0
    n_chasm = 0
    for cell in chasms:
        if not targets <= _reach(adj, bot, base_blocked | {cell}):
            extra += _chasm_open_extra(env, cell)
            n_chasm += 1
    for cell in rubble:
        if not targets <= _reach(adj, bot, base_blocked | {cell}):
            extra += env.COST_DRILL
    return True, extra, n_chasm


def h2_advanced(state: DeepCoreState, env: DeepCoreEnv) -> float:
    """
    Strengthens H1 while staying admissible and consistent (H1 <= H2 <= h*).

    1. Wall-aware geodesic distances instead of Manhattan, using the cheapest base cost per
       cell (sludge = 5). MST edge weight between two targets is min(d(u->v), d(v->u)); the
       bot's own leg is the directed distance to the nearest target.
    2. Payload term uses wall-aware *step* counts (the payload surcharge is paid per action).
    3. Incremental barrier cost: essential chasms (extra PUSH cost) and essential rubble
       piles (DRILL = 10) are added once each - never the move that the distance already pays.
    4. Deadlock pruning: returns inf if the state is provably unsolvable.
    """
    if env.is_goal(state):
        return 0.0

    steps = _get_wall_aware_distances(env)
    wcost = _get_weighted_distances(env)
    cache = _env_cache(env)

    bot = state.bot_pos
    elev = env.elevator_pos

    # ---- deadlock / barrier analysis (cached) ----
    dead = frozenset(b for b in state.boulders if _is_corner_deadlock(b, env))
    key = (bot, state.chasms, state.rubble, state.miners, dead)
    barrier_cache = cache.setdefault("barrier", {})
    info = barrier_cache.get(key)
    if info is None:
        info = _barrier_analysis(env, bot, state.chasms, state.rubble, state.miners, dead)
        barrier_cache[key] = info
    feasible, barrier_extra, n_essential_chasms = info
    if not feasible:
        return INF
    if n_essential_chasms > len(state.boulders) - len(dead):
        return INF  # not enough movable boulders left to bridge the required chasms

    # ---- MST over the targets + directed leg bot -> nearest target ----
    def edge(u: Pos, v: Pos) -> float:
        return min(wcost.get((u, v), INF), wcost.get((v, u), INF))

    targets = list({elev, *state.miners})
    mst = _prim_mst_cost(targets, edge)
    nearest = min(wcost.get((bot, t), INF) for t in targets)
    if mst == INF or nearest == INF:
        return INF

    # ---- payload surcharge: 2 per carried miner per remaining move/push ----
    payload = 2 * state.carried * steps.get((bot, elev), 0)
    payload += 2 * sum(steps.get((m, elev), 0) for m in state.miners)

    return float(mst + nearest + payload + barrier_extra)
