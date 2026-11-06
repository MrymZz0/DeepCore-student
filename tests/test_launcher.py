"""Tests for the zero-setup student launcher at the repository root."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    spec = importlib.util.spec_from_file_location("student_launcher", ROOT / "main.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStudentLauncher(unittest.TestCase):
    def test_root_launcher_accepts_no_arguments(self) -> None:
        launcher = _load_launcher()
        args = launcher.build_arg_parser().parse_args([])
        self.assertEqual(args.algo, "welcome")

    def test_root_launcher_keeps_cli_algorithm_options(self) -> None:
        launcher = _load_launcher()
        args = launcher.build_arg_parser().parse_args(["--algo", "astar", "--map", "level2"])
        self.assertEqual((args.algo, args.map), ("astar", "level2"))
