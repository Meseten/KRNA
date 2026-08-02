"""
KRNA / SKROA Operator Verification Suite
==============================================================================
Unit tests for mathematical invariants, vectorized array shapes, boundary
conditions, and numerical stability across Lévy Flight, Sympodial Clamping,
and Culm-Abortion operators.
"""

from __future__ import annotations
import unittest
import numpy as np

from krna.operators import (
    levy_flight_step,
    apply_sympodial_clamping,
    apply_culm_abortion
)


class TestLevyFlightStep(unittest.TestCase):
    """
    Verifies Mantegna's algorithm for Lévy Flight step generation.
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=101)

    def test_output_shape_and_finite_values(self) -> None:
        """
        Ensures output tensor matches requested (N, D) shape and contains no NaN/Inf.
        """
        shape = (50, 10)
        steps = levy_flight_step(shape=shape, beta=1.5, scale=0.1, rng=self.rng)
        
        self.assertEqual(steps.shape, shape)
        self.assertTrue(np.all(np.isfinite(steps)), "Lévy steps contain NaN or Inf values.")

    def test_beta_parameter_validation(self) -> None:
        """
        Ensures invalid stability parameter beta <= 0 or beta > 2 raises ValueError.
        """
        with self.assertRaises(ValueError):
            levy_flight_step(shape=(10, 2), beta=0.0, rng=self.rng)
            
        with self.assertRaises(ValueError):
            levy_flight_step(shape=(10, 2), beta=2.5, rng=self.rng)

    def test_scale_parameter_proportionality(self) -> None:
        """
        Ensures scaling factor linearly scales step magnitudes.
        """
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=42)
        
        step_base = levy_flight_step(shape=(20, 5), beta=1.5, scale=1.0, rng=rng1)
        step_scaled = levy_flight_step(shape=(20, 5), beta=1.5, scale=3.5, rng=rng2)
        
        np.testing.assert_allclose(
            step_scaled,
            step_base * 3.5,
            err_msg="Lévy flight step did not scale linearly with scale parameter."
        )


class TestSympodialClamping(unittest.TestCase):
    """
    Verifies anti-collision Sympodial Clamping (repulsion vector) invariants.
    """

    def test_repulsion_only_affects_weaker_agents(self) -> None:
        """
        When two agents are closer than epsilon_clamp, only the weaker agent
        (higher fitness in minimization) should be displaced away from the stronger.
        """
        # Two agents in 2D space, distance = 0.5
        positions = np.array([
            [0.0, 0.0],  # Agent 0: Stronger (f=0.0)
            [0.5, 0.0]   # Agent 1: Weaker   (f=10.0)
        ], dtype=np.float64)
        
        fitness = np.array([0.0, 10.0], dtype=np.float64)
        epsilon_clamp = 1.0  # Threshold > distance (0.5)
        eta = 0.1
        
        updated_positions = apply_sympodial_clamping(
            positions=positions,
            fitness=fitness,
            epsilon_clamp=epsilon_clamp,
            eta=eta
        )
        
        # 1. Stronger agent (Agent 0) must remain at exact original coordinates
        np.testing.assert_array_equal(
            updated_positions[0],
            positions[0],
            err_msg="Stronger agent position was incorrectly altered by clamping."
        )
        
        # 2. Weaker agent (Agent 1) must move further away along the x-axis (> 0.5)
        self.assertGreater(
            updated_positions[1, 0],
            0.5,
            msg="Weaker agent was not repulsed away from stronger neighbor."
        )

    def test_no_repulsion_outside_epsilon_threshold(self) -> None:
        """
        Agents separated by Euclidean distance >= epsilon_clamp must not interact.
        """
        positions = np.array([
            [0.0, 0.0],
            [5.0, 0.0]
        ], dtype=np.float64)
        
        fitness = np.array([1.0, 2.0], dtype=np.float64)
        epsilon_clamp = 1.0  # Threshold < distance (5.0)
        
        updated_positions = apply_sympodial_clamping(
            positions=positions,
            fitness=fitness,
            epsilon_clamp=epsilon_clamp
        )
        
        np.testing.assert_array_equal(
            updated_positions,
            positions,
            err_msg="Agents outside epsilon_clamp threshold were incorrectly displaced."
        )

    def test_single_agent_edge_case(self) -> None:
        """
        Swarm of size N=1 must return an identical copy without indexing errors.
        """
        positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        fitness = np.array([0.5], dtype=np.float64)
        
        updated = apply_sympodial_clamping(positions, fitness, epsilon_clamp=1.0)
        np.testing.assert_array_equal(updated, positions)

    def test_bounds_clipping(self) -> None:
        """
        Repulsed coordinates must be strictly clamped within search space bounds.
        """
        positions = np.array([
            [0.0, 0.0],
            [4.9, 0.0]   # Weaker agent near upper bound (5.0)
        ], dtype=np.float64)
        fitness = np.array([0.0, 10.0], dtype=np.float64)
        
        updated = apply_sympodial_clamping(
            positions=positions,
            fitness=fitness,
            epsilon_clamp=10.0,
            eta=5.0,
            bounds=(-5.0, 5.0)
        )
        
        self.assertLessEqual(updated[1, 0], 5.0, "Weaker agent exceeded upper bound after repulsion.")


class TestCulmAbortion(unittest.TestCase):
    """
    Verifies Culm-Abortion dynamic memory pruning and state reset invariants.
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(seed=202)
        self.bounds = (-10.0, 10.0)
        self.g_best = np.array([0.0, 0.0], dtype=np.float64)

    def test_stagnation_counter_increment_and_reset(self) -> None:
        """
        State 1 agents below tau_stagnation must increment counters.
        State 1 agents making progress must reset counters to 0.
        State 0 agents must always have counters reset to 0.
        """
        positions = np.zeros((3, 2), dtype=np.float64)
        previous_fitness = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        
        # Agent 0: State 1, stagnant (delta_f = 0.00001 < 0.001) -> counter ++
        # Agent 1: State 1, progress (delta_f = 1.0 >= 0.001)     -> counter = 0
        # Agent 2: State 0, stagnant (delta_f = 0.00001 < 0.001) -> counter = 0 (State 0 ignored)
        current_fitness = np.array([9.99999, 9.0, 9.99999], dtype=np.float64)
        states = np.array([1, 1, 0], dtype=int)
        stagnation_counters = np.array([2, 3, 5], dtype=int)
        
        _, _, new_counters, aborted_mask = apply_culm_abortion(
            positions=positions,
            current_fitness=current_fitness,
            previous_fitness=previous_fitness,
            states=states,
            stagnation_counters=stagnation_counters,
            global_best_position=self.g_best,
            tau_stagnation=1e-3,
            max_stagnation_steps=10,
            bounds=self.bounds,
            rng=self.rng
        )
        
        np.testing.assert_array_equal(new_counters, np.array([3, 0, 0]))
        self.assertFalse(np.any(aborted_mask), "No agent should be aborted before max_stagnation_steps.")

    def test_abortion_trigger_and_reallocation(self) -> None:
        """
        When stagnation counter hits max_stagnation_steps:
          1. State must reset from 1 to 0.
          2. Counter must reset to 0.
          3. Coordinates must be reallocated within search bounds.
          4. Aborted mask must flag True for the aborted agent.
        """
        positions = np.array([
            [5.0, 5.0],   # Agent 0: Will abort
            [-2.0, -2.0]  # Agent 1: Will remain
        ], dtype=np.float64)
        
        current_fitness = np.array([10.0, 10.0], dtype=np.float64)
        previous_fitness = np.array([10.0, 10.0], dtype=np.float64)  # 0 improvement
        states = np.array([1, 1], dtype=int)
        stagnation_counters = np.array([4, 1], dtype=int)  # Max is 5
        
        new_pos, new_states, new_counters, aborted_mask = apply_culm_abortion(
            positions=positions,
            current_fitness=current_fitness,
            previous_fitness=previous_fitness,
            states=states,
            stagnation_counters=stagnation_counters,
            global_best_position=self.g_best,
            tau_stagnation=1e-2,
            max_stagnation_steps=5,
            bounds=self.bounds,
            rng=self.rng
        )
        
        # Agent 0 hit counter = 5 -> Aborted
        self.assertTrue(aborted_mask[0], "Agent 0 was not flagged as aborted.")
        self.assertFalse(aborted_mask[1], "Agent 1 was incorrectly flagged as aborted.")
        
        # State and counter reset check
        self.assertEqual(new_states[0], 0, "Aborted agent state did not reset to 0.")
        self.assertEqual(new_counters[0], 0, "Aborted agent stagnation counter did not reset to 0.")
        
        # Position reallocation check (should no longer be [5.0, 5.0])
        self.assertFalse(
            np.allclose(new_pos[0], [5.0, 5.0]),
            "Aborted agent position was not reallocated."
        )
        # Check coordinates stay within bounds
        self.assertTrue(np.all(new_pos[0] >= self.bounds[0]) and np.all(new_pos[0] <= self.bounds[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)