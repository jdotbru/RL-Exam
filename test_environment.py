import math
import unittest
from unittest.mock import patch

import numpy as np

import fingerTriangleEnv as env_module
from fingerTriangleEnv import FingerTriangleEnv
from env_helpers import calculate_new_position, calculate_triangle_area


#Tests für die Funktionalität der Environment
class FingerTriangleEnvTests(unittest.TestCase):
    def make_env(self, use_antagonist: bool = False) -> FingerTriangleEnv:
        env = FingerTriangleEnv(givenUseAntagonist=use_antagonist)
        env.reset(seed=0)
        return env

    def test_reset_returns_valid_observation(self) -> None:
        env = FingerTriangleEnv()
        obs, info = env.reset(seed=123)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(np.allclose(env.currPos, env.startPos))
        self.assertEqual(len(env.positionSaver), 1)
        self.assertEqual(info["area"], 0.0)
        self.assertFalse(info["terminated"])
        self.assertFalse(info["truncated"])

    def test_update_angles_regular_action_uses_one_degree(self) -> None:
        #Test, ob Aktualisierung aufgrund einer Action korrekt funktioniert
        env = self.make_env()
        env.joint_angles = np.zeros(3, dtype=np.float32)

        env.updateAngles(26, antagonist=False)

        self.assertTrue(np.allclose(env.joint_angles, np.array([1.0, 1.0, 1.0], dtype=np.float32)))
        self.assertEqual(env.action_delta, 1.0)

    def test_update_angles_antagonist_uses_small_step(self) -> None:
        #Test, ob Antagonist-Aktion nur eine kleine Änderung vornimmt
        env = self.make_env()
        env.joint_angles = np.zeros(3, dtype=np.float32)

        env.updateAngles(26, antagonist=True)

        self.assertTrue(np.allclose(env.joint_angles, np.array([0.2, 0.2, 0.2], dtype=np.float32)))
        self.assertEqual(env.action_delta, 0.2)

    def test_step_with_zero_action_keeps_position_and_only_applies_step_penalty_in_phase0(self) -> None:
        #aktion 0 funktioniert korrekt in Aktualisierung der Gelenke und im Reward
        env = self.make_env()
        env.currPhase = 0
        env.step_ctr = 0
        env.joint_angles = np.zeros(3, dtype=np.float32)
        env.currPos = calculate_new_position(env.joint_angles, env.link_lengths)
        env.startPos = env.currPos.copy()
        env.positionSaver = [env.currPos.copy()]

        with patch.object(env_module, "is_corner", return_value=False):
            obs, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(np.allclose(env.currPos, env.startPos))
        self.assertEqual(env.step_ctr, 1)
        self.assertEqual(len(env.positionSaver), 2)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -env.penalty_stepNumberMultiplicator, places=6)
        self.assertAlmostEqual(info["reward"], reward, places=6)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_step_clips_joint_limits(self) -> None:
        #Testen, ob Gelenküberdrehung erkannt wird
        env = self.make_env()
        env.joint_angles = np.array([89.5, 89.5, 89.5], dtype=np.float32)
        env.currPos = calculate_new_position(env.joint_angles, env.link_lengths)
        env.startPos = env.currPos.copy()
        env.positionSaver = [env.currPos.copy()]

        with patch.object(env_module, "is_corner", return_value=False):
            env.step(26)

        self.assertTrue(np.all(env.joint_angles <= env.degree_limit))
        self.assertTrue(np.all(env.joint_angles >= -env.degree_limit))

    def test_curriculum_stage_updates_key_parameters(self) -> None:
        #Test, ob curriculum Stages wirlich die Parameter verändern
        env = self.make_env()

        env.set_curriculum_stage(0)
        stage0 = (
            env.closureRadius,
            env.minArea,
            env.penalty_ExtraCorner,
            env.penalty_Phase2ReturnLineDeviation,
        )

        env.set_curriculum_stage(3)
        stage3 = (
            env.closureRadius,
            env.minArea,
            env.penalty_ExtraCorner,
            env.penalty_Phase2ReturnLineDeviation,
        )

        self.assertGreater(stage0[0], stage3[0])
        self.assertLess(stage0[1], stage3[1])
        self.assertLess(stage0[2], stage3[2])
        self.assertGreater(stage3[3], 0.0)

    def test_is_finished_detects_valid_terminal_triangle(self) -> None:
        #Environment erkennt korrektes Dreieck
        env = self.make_env()
        env.set_curriculum_stage(0)
        env.currPhase = 2
        env.step_ctr = env.minSteps + 1
        env.startPos = np.array([0.0, 0.0], dtype=np.float32)
        env.corner1 = np.array([1.0, 0.0], dtype=np.float32)
        env.corner2 = np.array([0.0, 1.0], dtype=np.float32)
        env.currPos = np.array([0.1, 0.1], dtype=np.float32)

        self.assertGreaterEqual(calculate_triangle_area(env.startPos, env.corner1, env.corner2), env.minArea)
        self.assertTrue(env.isFinished())

    def test_is_truncated_at_max_steps(self) -> None:
        #Test, ob Truncation funktioniert
        env = self.make_env()
        env.step_ctr = math.ceil(env.maxSteps)

        self.assertTrue(env.isTruncated())

    def test_phase2_reward_increases_when_moving_towards_start(self) -> None:
        #Test, ob reward in phase 2 korrekt funktioniert
        env = self.make_env()
        env.set_curriculum_stage(3)
        env.currPhase = 2
        env.step_ctr = env.minSteps - 1
        env.lastCornerSteps = 0
        env.startPos = np.array([0.0, 0.0], dtype=np.float32)
        env.corner1 = np.array([3.0, 2.0], dtype=np.float32)
        env.corner2 = np.array([3.0, 0.0], dtype=np.float32)
        env.corner1_idx = 0
        env.corner2_idx = 0
        env.joint_angles = np.zeros(3, dtype=np.float32)
        env.currPos = np.array([2.4, 0.0], dtype=np.float32)
        env.positionSaver = [env.currPos.copy()]
        env.reward_Phase2CleanReturn = 0.0
        env.penalty_Phase2ReturnLineDeviation = 0.0
        env.penalty_Phase2LateDistance = 0.0
        env.useMaxSteps = False

        next_pos = np.array([1.9, 0.0], dtype=np.float32)
        expected_delta = np.linalg.norm(env.currPos - env.startPos) - np.linalg.norm(next_pos - env.startPos)
        expected_reward = env.reward_Phase2Closure * expected_delta - env.penalty_stepNumberMultiplicator

        with patch.object(env_module, "calculate_new_position", return_value=next_pos), patch.object(env_module, "is_corner", return_value=False):
            _, reward, terminated, truncated, _ = env.step(0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, expected_reward, places=6)


if __name__ == "__main__":
    print("Running environment tests...\n")
    unittest.main(verbosity=2)
