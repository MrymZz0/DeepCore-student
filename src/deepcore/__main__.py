"""
main.py
Unified Launcher for DeepCore AI Search Command Deck (GUI + CLI Benchmark Suite).
Covers all Core & Bonus Algorithms from the Course Syllabus:
  Core : BFS, DFS, IDS, Bidirectional, UCS, Greedy Best-First, A*, H1 vs H2
  Bonus: Weighted A*, IDA*, SMA*
[TA PROVIDED - DO NOT MODIFY]
"""

import argparse
import time
from typing import Callable, Dict, List, Tuple
from .environment import DeepCoreEnv, BUILTIN_MAPS
from .state import DeepCoreState, SearchMetrics
from . import solver, heuristics


HEURISTIC_REGISTRY: Dict[str, Callable[[DeepCoreState, DeepCoreEnv], float]] = {
    "naive": heuristics.h_naive_manhattan,
    "h1": heuristics.h1_basic,
    "h2": heuristics.h2_advanced,
}


def verify_solution(env: DeepCoreEnv, metrics: SearchMetrics) -> Tuple[bool, str]:
    if not metrics.solved:
        return False, getattr(metrics, "status_message", "No solution returned.")

    state = env.get_initial_state()
    recomputed_cost = 0
    try:
        for step_idx, act_str in enumerate(metrics.solution_actions, start=1):
            state, step_cost = env.transition(state, act_str)
            recomputed_cost += step_cost
    except Exception as err:
        return False, f"Illegal action at step {step_idx} ({act_str}): {err}"

    if not env.is_goal(state):
        return False, "Final state does not satisfy Goal Test G(s)!"

    metrics.total_cost = recomputed_cost
    return True, "Verified Optimal/Valid Solution"


def run_single_algorithm(
    env: DeepCoreEnv, algo: str, heur_name: str = "h2", weight: float = 1.5
) -> SearchMetrics:
    heur_fn = HEURISTIC_REGISTRY.get(heur_name, heuristics.h2_advanced)
    label_map = {
        "bfs": "BFS",
        "dfs": "DFS",
        "ids": "IDS",
        "bidirectional": "Bidirectional",
        "ucs": "UCS",
        "gbfs": f"GBFS({heur_name})",
        "astar": f"A*({heur_name})",
        "wastar": f"WA*({heur_name},W={weight:g})",
        "idastar": f"IDA*({heur_name})",
        "sma": f"SMA*({heur_name})",
    }
    label = label_map.get(algo, algo.upper())
    t0 = time.perf_counter()

    try:
        if algo == "bfs":
            metrics = solver.breadth_first_search(env)
        elif algo == "dfs":
            metrics = solver.depth_first_search(env)
        elif algo == "ids":
            metrics = solver.iterative_deepening_search(env)
        elif algo == "bidirectional":
            metrics = solver.bidirectional_search(env)
        elif algo == "ucs":
            metrics = solver.uniform_cost_search(env)
        elif algo == "gbfs":
            metrics = solver.greedy_best_first_search(env, heur_fn)
        elif algo == "astar":
            metrics = solver.a_star_search(env, heur_fn)
        elif algo == "wastar":
            metrics = solver.weighted_a_star_search(env, heur_fn, weight=weight)
        elif algo == "idastar":
            metrics = solver.ida_star_search(env, heur_fn)
        elif algo == "sma":
            metrics = solver.sma_star_search(env, heur_fn)
        else:
            raise ValueError(f"Unknown algorithm '{algo}'")
    except NotImplementedError:
        elapsed = time.perf_counter() - t0
        return SearchMetrics(
            algorithm_name=label,
            solved=False,
            runtime_sec=elapsed,
            status_message=f"{label} is not implemented yet.",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return SearchMetrics(
            algorithm_name=label,
            solved=False,
            runtime_sec=elapsed,
            status_message=f"Runtime Error in {label}: {exc}",
        )

    elapsed = time.perf_counter() - t0
    if metrics is None:
        return SearchMetrics(
            algorithm_name=label,
            solved=False,
            runtime_sec=elapsed,
            status_message=f"{label} returned None.",
        )

    metrics.algorithm_name = label
    metrics.runtime_sec = elapsed
    valid, msg = verify_solution(env, metrics)
    metrics.solved = valid
    metrics.status_message = msg
    return metrics


def print_cli_benchmark(levels: List[str], weight: float = 1.5) -> None:
    suite = [
        ("bfs", "naive", 1.0),
        ("dfs", "naive", 1.0),
        ("ids", "naive", 1.0),
        ("bidirectional", "naive", 1.0),
        ("ucs", "naive", 1.0),
        ("gbfs", "h2", 1.0),
        ("astar", "naive", 1.0),
        ("astar", "h1", 1.0),
        ("astar", "h2", 1.0),
        ("wastar", "h2", weight),
        ("idastar", "h2", 1.0),
        ("sma", "h2", 1.0),
    ]
    for lvl in levels:
        env = DeepCoreEnv.from_file_or_builtin(lvl)
        print("\n" + "=" * 88)
        print(f" DEEPCORE SYLLABUS BENCHMARK — {env.level_name} ({env.rows}x{env.cols}, Miners: {env.total_miners})")
        print("=" * 88)
        print(f"{'Algorithm':<18} | {'Solved':<6} | {'Steps':>5} | {'Cost':>6} | {'Expanded':>9} | {'Generated':>9} | {'Frontier':>8} | {'Memory':>8} | {'Time (ms)':>9}")
        print("-" * 116)
        for algo, heur, w in suite:
            m = run_single_algorithm(env, algo, heur, w)
            steps_s = str(len(m.solution_actions)) if m.solved else "-"
            cost_s = str(m.total_cost) if m.solved else "-"
            print(
                f"{m.algorithm_name:<18} | {str(m.solved):<6} | {steps_s:>5} | "
                f"{cost_s:>6} | {m.nodes_expanded:>9} | {m.nodes_generated:>9} | "
                f"{m.max_frontier_size:>8} | {m.max_memory_size:>8} | "
                f"{m.runtime_sec * 1000:>9.2f}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepCore AI Search Command Deck")
    parser.add_argument("-m", "--map", type=str, default="level1", help="Map level ('level1'..'level5' or 'all').")
    parser.add_argument(
        "-a", "--algo",
        type=str,
        default="manual",
        choices=["manual", "bfs", "dfs", "ids", "bidirectional", "ucs", "gbfs", "astar", "wastar", "idastar", "sma"],
        help="Algorithm to run or 'manual' to play interactively.",
    )
    parser.add_argument("-H", "--heuristic", type=str, default="h2", choices=["naive", "h1", "h2"])
    parser.add_argument("-w", "--weight", type=float, default=1.5, help="Weight W for Weighted A* (default: 1.5).")
    parser.add_argument("--cli", action="store_true", help="Run in headless terminal mode.")
    parser.add_argument("--benchmark", action="store_true", help="Print full syllabus benchmark table in terminal.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.benchmark:
        target_levels = list(BUILTIN_MAPS.keys()) if args.map.lower() == "all" else [args.map]
        print_cli_benchmark(target_levels, args.weight)
        return

    if args.cli:
        env = DeepCoreEnv.from_file_or_builtin(args.map)
        m = run_single_algorithm(env, args.algo, args.heuristic, args.weight)
        print(f"Level          : {env.level_name}")
        print(f"Algorithm      : {m.algorithm_name}")
        print(f"Solved         : {m.solved} ({m.status_message})")
        print(f"Steps          : {len(m.solution_actions)}")
        print(f"Total Cost     : {m.total_cost}")
        print(f"Nodes Expanded : {m.nodes_expanded}")
        print(f"Nodes Generated: {m.nodes_generated}")
        print(f"Max Frontier   : {m.max_frontier_size}")
        print(f"Max Memory     : {m.max_memory_size}")
        print(f"Runtime        : {m.runtime_sec * 1000:.2f} ms")
        return

    from .visualizer import DeepCoreWorkbench
    workbench = DeepCoreWorkbench(
        initial_level=args.map,
        initial_algo=args.algo,
        initial_heur=args.heuristic,
        initial_weight=args.weight,
        runner_callback=run_single_algorithm,
        heuristic_lookup=HEURISTIC_REGISTRY,
    )
    workbench.run()


if __name__ == "__main__":
    main()
