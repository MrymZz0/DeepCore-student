"""The student-friendly DeepCore launcher.

Run ``python main.py`` from the extracted assignment folder.  It works before
package installation because it adds the local ``src`` directory to Python's
import path.  Advanced command-line options are forwarded to ``deepcore``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REQUIREMENTS = ROOT / "requirements.txt"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_requirements() -> None:
    """Install the project's dependencies before starting the student app."""
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing dependency file: {REQUIREMENTS}")

    print("Checking and installing project requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
        check=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the small, discoverable command line used by the root launcher."""
    parser = argparse.ArgumentParser(description="DeepCore student launcher")
    parser.add_argument("--map", default="level1", help="Map name: level1 through level5.")
    parser.add_argument(
        "--algo",
        default="welcome",
        choices=["welcome", "manual", "bfs", "dfs", "ids", "bidirectional", "ucs", "gbfs", "astar", "wastar", "idastar", "sma"],
        help="Choose an algorithm, or omit this option to open the starter menu.",
    )
    parser.add_argument("--heuristic", default="h2", choices=["naive", "h1", "h2"])
    parser.add_argument("--weight", type=float, default=1.5)
    parser.add_argument("--benchmark", action="store_true", help="Run the benchmark table.")
    parser.add_argument("--gui", action="store_true", help="Open the graphical workbench.")
    return parser


def _run_cli_demo() -> None:
    from deepcore.__main__ import run_single_algorithm
    from deepcore.environment import DeepCoreEnv

    env = DeepCoreEnv.from_file_or_builtin("level1")
    result = run_single_algorithm(env, "bfs", "naive")
    print(f"\nLevel: {env.level_name} | Algorithm: {result.algorithm_name}")
    print(f"Solved: {result.solved} | Steps: {len(result.solution_actions)} | Cost: {result.total_cost}")
    print("\nNext: implement a TODO, then run option 2 to check your work.")


def _run_tests() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    print(f"\nTests finished with exit code {completed.returncode}.")


def _open_gui() -> None:
    try:
        from deepcore.__main__ import main as package_main
        import pygame  # noqa: F401  # Gives a clear setup message before opening the UI.
    except ModuleNotFoundError:
        print("The visual interface needs pygame-ce. Run python main.py again to install it.")
        return
    old_argv = sys.argv
    try:
        sys.argv = ["deepcore"]
        package_main()
    finally:
        sys.argv = old_argv


def show_welcome_menu() -> None:
    """Offer first-time students useful actions without requiring arguments."""
    print("\n" + "=" * 58)
    print("DeepCore - AI Search Assignment")
    print("=" * 58)
    print("1. Run the provided Level 1 BFS demonstration")
    print("2. Run the public tests")
    print("3. Open the visual workbench (pygame required)")
    print("4. Read the student guide")
    print("0. Exit")
    choice = input("Choose an option [1]: ").strip() or "1"
    if choice == "1":
        _run_cli_demo()
    elif choice == "2":
        _run_tests()
    elif choice == "3":
        _open_gui()
    elif choice == "4":
        print("Open docs/student-guide-fa.md, then read docs/assignment.html.")
    elif choice != "0":
        print("Please run python main.py again and select 0, 1, 2, 3, or 4.")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _install_requirements()
    if args.algo == "welcome" and not args.benchmark and not args.gui:
        show_welcome_menu()
        return
    if args.gui:
        _open_gui()
        return

    # Keep package and root command behaviour aligned without duplicating the
    # algorithm dispatcher in two places.
    from deepcore.__main__ import main as package_main

    forwarded = list(argv) if argv is not None else sys.argv[1:]
    old_argv = sys.argv
    try:
        sys.argv = ["deepcore", "--cli", *forwarded]
        package_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
