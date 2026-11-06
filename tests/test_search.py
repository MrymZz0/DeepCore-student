"""
test_search.py
Student Unit Test Suite for DeepCore Search Algorithms & Heuristics.

Run with:
    python test_search.py

Tests 1 and 2 pass with the supplied reference examples. Tests 3–7 define the
public acceptance criteria for the algorithms and heuristics you implement.
"""

import unittest
from deepcore.environment import DeepCoreEnv
from deepcore import solver, heuristics


class TestDeepCoreSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.env1 = DeepCoreEnv.from_file_or_builtin("level1")
        self.env2 = DeepCoreEnv.from_file_or_builtin("level2")
        self.env3 = DeepCoreEnv.from_file_or_builtin("level3")

    # =========================================================================
    # TEST 1 [PROVIDED]: Verify BFS solves Level 1 and minimizes step count
    # =========================================================================
    def test_01_bfs_solves_level1(self) -> None:
        metrics = solver.breadth_first_search(self.env1)
        self.assertIsNotNone(metrics)
        self.assertTrue(metrics.solved, "BFS should solve Level 1.")
        # Verify by replaying actions
        state = self.env1.get_initial_state()
        for act in metrics.solution_actions:
            state, _ = self.env1.transition(state, act)
        self.assertTrue(self.env1.is_goal(state))
        self.assertEqual(len(metrics.solution_actions), 22, "BFS should find the 22-step path on Level 1.")

    # =========================================================================
    # TEST 2 [PROVIDED]: Verify Naive Heuristic H0 is 0 at Goal and non-negative
    # =========================================================================
    def test_02_naive_heuristic_baseline(self) -> None:
        s0 = self.env1.get_initial_state()
        h_val = heuristics.h_naive_manhattan(s0, self.env1)
        self.assertGreater(h_val, 0.0)
        for goal_state in self.env1.get_goal_states():
            self.assertEqual(heuristics.h_naive_manhattan(goal_state, self.env1), 0.0)

    # =========================================================================
    # TEST 3 [PUBLIC]: UCS finds a strictly cheaper cost than BFS
    # =========================================================================
    def test_03_ucs_beats_bfs_cost_on_level1(self) -> None:
        """
        On Level 1, BFS wades through the sludge lane: 22 steps, Cost = 74.
        UCS takes the longer dry/rubble lane: 28 steps, Optimal Cost = 68.
        Expected: metrics.solved is True and metrics.total_cost is 68,
        strictly lower than BFS's expected cost of 74.
        """
        bfs_metrics = solver.breadth_first_search(self.env1)
        ucs_metrics = solver.uniform_cost_search(self.env1)
        self.assertTrue(ucs_metrics.solved)
        self.assertEqual(ucs_metrics.total_cost, 68)
        self.assertLess(ucs_metrics.total_cost, bfs_metrics.total_cost)

    # =========================================================================
    # TEST 4 [PUBLIC]: Bidirectional Search & IDS on Level 1
    # =========================================================================
    def test_04_uninformed_ids_and_bidirectional(self) -> None:
        """
        Both algorithms must return a valid 22-step solution.
        """
        ids_m = solver.iterative_deepening_search(self.env1)
        bidir_m = solver.bidirectional_search(self.env1)
        self.assertTrue(ids_m.solved)
        self.assertTrue(bidir_m.solved)
        self.assertEqual(len(ids_m.solution_actions), 22)
        self.assertEqual(len(bidir_m.solution_actions), 22)

    # =========================================================================
    # TEST 5 [PUBLIC]: Heuristic Dominance & Admissibility (H0 <= H1 <= H2 <= h*)
    # =========================================================================
    def test_05_heuristic_dominance_and_admissibility(self) -> None:
        """
        Checks the initial state of Level 2 against the UCS optimal cost.
        """
        s0 = self.env2.get_initial_state()
        ucs_m = solver.uniform_cost_search(self.env2)
        h0 = heuristics.h_naive_manhattan(s0, self.env2)
        h1 = heuristics.h1_basic(s0, self.env2)
        h2 = heuristics.h2_advanced(s0, self.env2)

        self.assertLessEqual(h0, h1, "H1 must dominate H0.")
        self.assertLessEqual(h1, h2, "H2 must dominate H1.")
        self.assertLessEqual(h2, ucs_m.total_cost, "H2 must be admissible (never overestimate optimal cost).")

    # =========================================================================
    # TEST 6 [PUBLIC]: A* Optimality & Node Expansion Reduction
    # =========================================================================
    def test_06_astar_optimality_and_efficiency(self) -> None:
        """
        All variants must produce the same optimal cost, and H2 must expand
        fewer nodes than the naive Manhattan heuristic.
        """
        m_naive = solver.a_star_search(self.env3, heuristics.h_naive_manhattan)
        m_h1 = solver.a_star_search(self.env3, heuristics.h1_basic)
        m_h2 = solver.a_star_search(self.env3, heuristics.h2_advanced)

        self.assertTrue(m_naive.solved and m_h1.solved and m_h2.solved)
        self.assertEqual(m_h1.total_cost, m_naive.total_cost)
        self.assertEqual(m_h2.total_cost, m_naive.total_cost)
        self.assertLess(m_h2.nodes_expanded, m_naive.nodes_expanded)

    # =========================================================================
    # TEST 7 [PUBLIC]: Longer Maps Complete for Deep Searches
    # =========================================================================
    def test_07_longer_maps_complete_for_deep_searches(self) -> None:
        """Deep-search entry points must not report artificial budget exhaustion and must have valid answers."""
        for level in ("level2", "level3", "level4", "level5"):
            env = DeepCoreEnv.from_file_or_builtin(level)
            for metrics in (
                solver.iterative_deepening_search(env),
                solver.bidirectional_search(env),
                solver.ida_star_search(env, heuristics.h2_advanced),
            ):
                self.assertTrue(metrics.solved, f"{metrics.algorithm_name} failed on {level}")
                state = env.get_initial_state()
                for action in metrics.solution_actions:
                    state, _ = env.transition(state, action)
                self.assertTrue(env.is_goal(state), f"Invalid path on {level}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
