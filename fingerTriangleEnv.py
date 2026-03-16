from random import seed
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple, Optional
import math

class FingerTriangleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, givenUseAntagonist=False):
        super().__init__()
        
        #einstellbare Parameter
        self.useMaxSteps = True
        self.useAntagonist = bool(givenUseAntagonist)
        self.maxSteps = 140
        self.minSteps = 24
        self.angleThresholdMin = 12.0
        self.angleThresholdMax = 150.0
        self.minCornerSteps = 4
        self.min_segment_len = 0.2
        self.action_delta = 1.0
        self.closureRadius = 1.5
        self.closureRadiusMin = 0.5
        self.closureRadiusMax = 1.0
        self.closureRadiusRatio = 0.09
        self.cornerWindow = 3
        self.directionChangeLimit = 25.0
        self.minArea = 0.1
        self.targetAngle = 90.0
        self.phase0_soft_limit = 40
        self.phase1_soft_limit = 90
        self.maxMeanLineDeviation = 0.35
        self.minStraightnessForSuccess = 0.35
        self.maxExtraCornersForSuccess = 2
        self.minMeanEdgeLengthForGoodTriangle = 1.0
        self.minEdgeBalanceForGoodTriangle = 0.45
        self.curriculum_stage = 0
        
        #penalties and rewards
        self.totalReward = 0
        self.reward_Closure = 150.0
        self.reward_Corner1 = 25.0
        self.reward_Corner2 = 40.0
        self.reward_Multiplier_Area_Corner2 = 60.0
        self.reward_Multiplier_Area = 55.0
        self.reward_Multiplier_Area_InSequence = 75.0
        self.reward_partialArea = 12.0
        self.reward_Edge1Len = 0.8
        self.reward_Phase2Closure = 10.0
        self.reward_Phase2AreaPreservation = 0.0
        self.reward_TriangleShape = 120.0
        self.reward_TriangleStraightness = 110.0
        self.penalty_limitViolation = 0.05
        self.penalty_noTriangle = 15.0
        self.penalty_stepNumberMultiplicator = 0.01
        self.penalty_Multiplier_DirectionChange0 = 0.0015
        self.penalty_Multiplier_DirectionChange1 = 0.0008
        self.penalty_Phase2AwayFromStart = 8.0
        self.penalty_Phase2EndDistance = 4.0
        self.penalty_ExtraCorner = 16.0
        self.penalty_SegmentCurvature = 5.0
        self.penalty_phaseStall0 = 0.03
        self.penalty_phaseStall1 = 0.05
        self.phaseProgressScale0 = 0.05
        self.phaseProgressScale1 = 0.8
        self.phaseProgressScale2 = 10.0
        self.reward_Phase2DirectionAlignment = 2.0
        self.reward_Phase2CleanReturn = 0.0
        self.penalty_Phase2ReturnLineDeviation = 6.0
        self.penalty_Phase2LateDistance = 1.5
        self.stage2_maxCountedExtraCorners = 2
        self.stage2_curvaturePenaltyCap = 10.0
        self.stage2_awayFromStartPenaltyCap = 10.0
        self.stage2_lateDistancePenaltyCap = 8.0
        self.stage2_truncationEndDistancePenaltyCap = 8.0
        self.stage3_curvaturePenaltyCap = 8.0
        self.stage3_awayFromStartPenaltyCap = 8.0
        self.stage3_lateDistancePenaltyCap = 6.0
        self.stage3_truncationEndDistancePenaltyCap = 6.0
        self.reward_Phase1Corner2Spread = 9.0
        self.reward_TriangleMeanEdge = 20.0
        self.reward_TriangleEdgeBalance = 28.0
        self.penalty_TerminalGap = 10.0
        self.reward_TerminalEdge3Length = 6.0
    
        #gesetzte Parameter
        self.link_lengths = np.array([5.0, 2.5, 2.5])
        self.degree_limit = 90
        
        #Random Number Generator & State-Variablen
        self.rng = np.random.default_rng(0)
        self.step_ctr = 0
        self.joint_angles = np.zeros(3)
        self.currPos = np.zeros(2)
        self.startPos = np.zeros(2)
        self.positionSaver: list[np.ndarray] = []
        self.currPhase = 0
        self.corner1 = None
        self.corner2 = None
        self.corner1_idx = None
        self.corner2_idx = None
        self.lastCornerSteps = 0
        self.extraCornerCount = 0
        self.phase2_target_direction = None
        self.bestFormSnapshot = None
        self.bestFormScore = float("-inf")
        
        #Action-space
        #Diskret mit 9 Aktionen, 3 für jeden Winkel 
        if self.useAntagonist:
            self.action_space = spaces.Dict({
                #tabelle für Aktionen liegt im root-Verzeichnis
                "protagonist": spaces.Discrete(27),
                "antagonist": spaces.Discrete(27)
            })
        else:
            self.action_space = spaces.Discrete(27)
        
        #Observation-space
        #Grenzen des Fingers liegen bei -10:10 (vollausgestreckte addition aller Fingerteile), somit liegen die Grenzen der Differenz zwischen zwei Punkten bei -20:20
        obs_low = np.array(
            [-self.degree_limit] * 3
            + [-10] * 2
            + [-20] * 2
            + [-20] * 2
            + [-20] * 2
            + [-20] * 2
            + [-1] * 2
            + [-1] * 2
            + [0]
            + [0]
            + [0]
            + [-1] * 2
            + [0]
            + [0]
            + [0] * 3,
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.degree_limit] * 3
            + [10] * 2
            + [20] * 2
            + [20] * 2
            + [20] * 2
            + [20] * 2
            + [1] * 2
            + [1] * 2
            + [10]
            + [10]
            + [20]
            + [1] * 2
            + [20]
            + [20]
            + [1] * 3,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.set_curriculum_stage(0)
    
    #---------------------    
    #---Hilfsfunktionen---
    #---------------------
    def get_obs(self) -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        distancetoStart = (self.currPos - self.startPos).astype(np.float32)
        vectorToStart = (self.startPos - self.currPos).astype(np.float32)

        if self.corner1 is None:
            distanceToCorner1 = np.zeros(2, dtype=np.float32)
        else:
            distanceToCorner1 = (self.currPos - self.corner1).astype(np.float32)

        # Relative Position zu corner2
        if self.corner2 is None:
            distanceToCorner2 = np.zeros(2, dtype=np.float32)
            desiredReturnDirection = np.zeros(2, dtype=np.float32)
        else:
            distanceToCorner2 = (self.currPos - self.corner2).astype(np.float32)
            desiredReturnDirection = self.normalizeVector(self.startPos - self.corner2).astype(np.float32)

        currentMoveDirection = self.getCurrentMoveDirection().astype(np.float32)
        effectiveClosureRadius = np.array([self.getEffectiveClosureRadius()], dtype=np.float32)
        returnLineDeviation = np.array([self.getReturnLineDeviation()], dtype=np.float32)
        distanceToStartNorm = np.array([np.linalg.norm(self.currPos - self.startPos)], dtype=np.float32)
        phaseTargetDirection = self.getPhaseTargetDirection().astype(np.float32)
        phaseLineDeviation = np.array([self.getPhaseLineDeviation()], dtype=np.float32)
        phaseProgress = np.array([self.getPhaseProgress()], dtype=np.float32)
        
        phase_one_hot = np.zeros(3, dtype=np.float32)
        phase_idx = int(np.clip(self.currPhase, 0, 2))
        phase_one_hot[phase_idx] = 1.0
        return np.concatenate(
            [
                self.joint_angles,
                self.currPos,
                distancetoStart,
                distanceToCorner1,
                distanceToCorner2,
                vectorToStart,
                desiredReturnDirection,
                currentMoveDirection,
                effectiveClosureRadius,
                returnLineDeviation,
                distanceToStartNorm,
                phaseTargetDirection,
                phaseLineDeviation,
                phaseProgress,
                phase_one_hot,
            ]
        ).astype(np.float32)

    def set_curriculum_stage(self, stage: int) -> None:
        self.curriculum_stage = int(stage)

        if self.curriculum_stage <= 0:
            self.maxSteps = 130
            self.minSteps = 16
            self.closureRadius = 2.2
            self.minArea = 0.03
            self.maxMeanLineDeviation = 0.55
            self.minStraightnessForSuccess = 0.20
            self.maxExtraCornersForSuccess = 3
            self.minMeanEdgeLengthForGoodTriangle = 0.55
            self.minEdgeBalanceForGoodTriangle = 0.28
            self.penalty_ExtraCorner = 8.0
            self.penalty_SegmentCurvature = 2.5
            self.reward_TriangleStraightness = 70.0
            self.reward_Phase2AreaPreservation = 0.0
            self.reward_Phase2DirectionAlignment = 2.0
            self.reward_Phase2CleanReturn = 0.0
        elif self.curriculum_stage == 1:
            self.maxSteps = 142
            self.minSteps = 20
            self.closureRadius = 1.8
            self.minArea = 0.06
            self.maxMeanLineDeviation = 0.45
            self.minStraightnessForSuccess = 0.28
            self.maxExtraCornersForSuccess = 2
            self.minMeanEdgeLengthForGoodTriangle = 0.75
            self.minEdgeBalanceForGoodTriangle = 0.34
            self.penalty_ExtraCorner = 12.0
            self.penalty_SegmentCurvature = 4.0
            self.reward_TriangleStraightness = 95.0
            self.reward_Phase2AreaPreservation = 0.0
            self.reward_Phase2DirectionAlignment = 2.0
            self.reward_Phase2CleanReturn = 0.0
        elif self.curriculum_stage == 2:
            self.maxSteps = 148
            self.minSteps = 22
            self.closureRadius = 1.65
            self.minArea = 0.08
            self.maxMeanLineDeviation = 0.40
            self.minStraightnessForSuccess = 0.38
            self.maxExtraCornersForSuccess = 1
            self.minMeanEdgeLengthForGoodTriangle = 0.95
            self.minEdgeBalanceForGoodTriangle = 0.40
            self.penalty_ExtraCorner = 15.0
            self.penalty_SegmentCurvature = 5.0
            self.reward_TriangleStraightness = 108.0
            self.reward_Phase2AreaPreservation = 0.0
            self.reward_Phase2DirectionAlignment = 2.0
            self.reward_Phase2CleanReturn = 0.0
            self.penalty_Phase2AwayFromStart = 8.0
            self.penalty_Phase2ReturnLineDeviation = 6.0
            self.penalty_Phase2LateDistance = 1.5
        else:
            self.maxSteps = 156
            self.minSteps = 23
            self.closureRadius = 1.63
            self.minArea = 0.08
            self.maxMeanLineDeviation = 0.40
            self.minStraightnessForSuccess = 0.40
            self.maxExtraCornersForSuccess = 1
            self.minMeanEdgeLengthForGoodTriangle = 0.97
            self.minEdgeBalanceForGoodTriangle = 0.41
            self.penalty_ExtraCorner = 15.5
            self.penalty_SegmentCurvature = 5.0
            self.reward_TriangleStraightness = 110.0
            self.reward_Phase2Closure = 11.0
            self.reward_Phase2AreaPreservation = 0.0
            self.reward_Phase2DirectionAlignment = 2.2
            self.reward_Phase2CleanReturn = 3.0
            self.penalty_Phase2AwayFromStart = 7.0
            self.penalty_Phase2ReturnLineDeviation = 5.0
            self.penalty_Phase2LateDistance = 1.2
    
    def updateAngles(self, action): 
        
        #updating angle 1 
        if action > 8 and action <= 17:
            self.joint_angles[0] -= self.action_delta
        elif action > 17:
            self.joint_angles[0] += self.action_delta
        
        #updating angle 2
        if (action > 2 and action < 6) or (action > 11 and action < 15) or (action > 20 and action < 24):
            self.joint_angles[1] -= self.action_delta
        elif (action > 5 and action < 9) or (action > 14 and action < 18) or action > 23:
            self.joint_angles[1] += self.action_delta
        
        #update angle 3
        if (action)%3 == 1:
            self.joint_angles[2] -= self.action_delta
        elif (action)%3 == 2:
            self.joint_angles[2] += self.action_delta
            
    def angleBetween(self, v1: np.ndarray, v2: np.ndarray) -> float:
        #Berechnet den Winkel zwischen zwei Punkten
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm == 0:
            return 0.0
        
        cosAngle = np.clip(dot / norm, -1.0, 1.0)
        return np.degrees(np.arccos(cosAngle))
    
    def isCorner(self) -> bool:
        #Schaut, ob mit dem Aktuellen Schritt eine Richtungsänderung durchgeführt wurde, die stark genug ist um als Ecke erkannt zu werden
        if len(self.positionSaver) < 2 * self.cornerWindow + 1:
            return False
        
        p1 = self.positionSaver[-(2*self.cornerWindow+1)]
        p2 = self.positionSaver[-(self.cornerWindow+1)]
        p3 = self.positionSaver[-1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Mindestlänge der Segmente prüfen
        if np.linalg.norm(v1) < self.min_segment_len or np.linalg.norm(v2) < self.min_segment_len:
            return False
        
        angle = self.angleBetween(v1, v2)
        
        return self.angleThresholdMax > angle > self.angleThresholdMin
    
    #To-Do: Update Function to non-angle values
    def calculate_new_Position(self, angles: np.ndarray) -> np.ndarray:
        #berechnet neue Position des Fingers
        p1, p2, p3 = self.link_lengths
        a1, a2, a3 = np.deg2rad(angles)
        
        b1 = a1
        b2 = a1 + a2
        b3 = a1 + a2 + a3
        
        x = p1 * np.cos(b1) + p2 * np.cos(b2) + p3 * np.cos(b3)
        y = p1 * np.sin(b1) + p2 * np.sin(b2) + p3 * np.sin(b3)
        return np.array([x, y])
    
    def calculateTriangleArea(self, corner1: np.ndarray, corner2: np.ndarray, corner3: np.ndarray) -> float:
        #Flächenformel: 0,5 * |det(corner2-corner1, corner3-corner1)|     
        v1 = corner2 - corner1
        v2 = corner3 - corner1
        det = v1[0]*v2[1] - v1[1]*v2[0]
        return float(0.5 * abs(det))

    def getTriangleEdgeLengths(self) -> tuple[float, float, float]:
        if self.corner1 is None or self.corner2 is None:
            return 0.0, 0.0, 0.0
        edge1 = float(np.linalg.norm(self.corner1 - self.startPos))
        edge2 = float(np.linalg.norm(self.corner2 - self.corner1))
        edge3 = float(np.linalg.norm(self.startPos - self.corner2))
        return edge1, edge2, edge3

    def getTriangleEdgeMetrics(self) -> tuple[float, float]:
        edge1, edge2, edge3 = self.getTriangleEdgeLengths()
        edges = np.array([edge1, edge2, edge3], dtype=np.float32)
        mean_edge_length = float(np.mean(edges))
        max_edge_length = float(np.max(edges))
        if max_edge_length <= 1e-6:
            return mean_edge_length, 0.0
        edge_balance = float(np.min(edges) / max_edge_length)
        return mean_edge_length, edge_balance

    def getEdge1Quality(self) -> tuple[float, float]:
        if self.corner1 is None:
            return 0.0, 0.0
        edge1_len = float(np.linalg.norm(self.corner1 - self.startPos))
        if self.corner1_idx is None:
            return edge1_len, 0.0
        seg1 = self.getSegmentPoints(0, self.corner1_idx)
        deviation = self.segmentMeanDeviation(self.startPos, self.corner1, seg1)
        return edge1_len, float(deviation)

    def getEdge2Quality(self) -> tuple[float, float]:
        if self.corner1 is None or self.corner2 is None:
            return 0.0, 0.0
        edge2_len = float(np.linalg.norm(self.corner2 - self.corner1))
        if self.corner1_idx is None or self.corner2_idx is None:
            return edge2_len, 0.0
        seg2 = self.getSegmentPoints(self.corner1_idx, self.corner2_idx)
        deviation = self.segmentMeanDeviation(self.corner1, self.corner2, seg2)
        return edge2_len, float(deviation)

    def isGoodTriangle(self) -> bool:
        if self.corner1 is None or self.corner2 is None or self.currPhase != 2:
            return False

        area, _, straightness_score = self.evaluateTriangleShape()
        mean_edge_length, edge_balance = self.getTriangleEdgeMetrics()
        return bool(
            area >= self.minArea
            and straightness_score >= self.minStraightnessForSuccess
            and self.extraCornerCount <= self.maxExtraCornersForSuccess
            and mean_edge_length >= self.minMeanEdgeLengthForGoodTriangle
            and edge_balance >= self.minEdgeBalanceForGoodTriangle
        )

    def calculateTurnAngle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        
        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
            turnAngle = self.angleBetween(v1, v2)
        else:
            turnAngle = 0.0
        return turnAngle
    
    def calculateParallelity(self) -> float:
        edge1 = self.corner1 - self.startPos
        edge2 = self.currPos - self.corner1

        n1 = np.linalg.norm(edge1)
        n2 = np.linalg.norm(edge2)

        if n1 > 1e-6 and n2 > 1e-6:
            cos_sim = np.dot(edge1, edge2) / (n1 * n2)
        else:
            cos_sim = 0
        
        return cos_sim
    
    def calculateAngleOfEdge2(self) -> float:
        edge1 = self.corner1 - self.startPos
        edge2 = self.currPos - self.corner1

        angle = self.angleBetween(edge1, edge2)
        return angle

    def normalizeVector(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def getCurrentMoveDirection(self) -> np.ndarray:
        if len(self.positionSaver) < 2:
            return np.zeros(2, dtype=np.float32)
        return self.normalizeVector(self.positionSaver[-1] - self.positionSaver[-2])

    def maybeInitializePhaseTargetDirection(self) -> None:
        if self.currPhase == 2 and self.corner2 is not None and self.phase2_target_direction is None:
            direction = self.startPos - self.corner2
            if np.linalg.norm(direction) >= self.min_segment_len:
                self.phase2_target_direction = self.normalizeVector(direction)

    def getPhaseAnchor(self) -> Optional[np.ndarray]:
        if self.currPhase == 2:
            return self.corner2
        return None

    def getPhaseTargetDirection(self) -> np.ndarray:
        if self.currPhase == 2:
            if self.phase2_target_direction is not None:
                return self.phase2_target_direction
            if self.corner2 is not None:
                return self.normalizeVector(self.startPos - self.corner2)
        return np.zeros(2, dtype=np.float32)

    def getPhaseLineDeviation(self, position: Optional[np.ndarray] = None) -> float:
        anchor = self.getPhaseAnchor()
        direction = self.getPhaseTargetDirection()
        if anchor is None or np.linalg.norm(direction) < 1e-6:
            return 0.0
        position = self.currPos if position is None else position
        line_end = anchor + 20.0 * direction
        return self.pointLineDistance(position, anchor, line_end)

    def getPhaseProgress(self, position: Optional[np.ndarray] = None) -> float:
        anchor = self.getPhaseAnchor()
        direction = self.getPhaseTargetDirection()
        if anchor is None or np.linalg.norm(direction) < 1e-6:
            return 0.0
        position = self.currPos if position is None else position
        return float(max(0.0, np.dot(position - anchor, direction)))

    def pointLineDistance(self, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        segment = end - start
        seg_norm = np.linalg.norm(segment)
        if seg_norm < 1e-6:
            return float(np.linalg.norm(point - start))

        rel = point - start
        projection = np.dot(rel, segment) / (seg_norm ** 2)
        projection = np.clip(projection, 0.0, 1.0)
        closest = start + projection * segment
        return float(np.linalg.norm(point - closest))

    def segmentMeanDeviation(self, start: np.ndarray, end: np.ndarray, points: list[np.ndarray]) -> float:
        if len(points) <= 2:
            return 0.0
        distances = [self.pointLineDistance(point, start, end) for point in points[1:-1]]
        if not distances:
            return 0.0
        return float(np.mean(distances))

    def getSegmentPoints(self, start_idx: int, end_idx: int) -> list[np.ndarray]:
        return [np.asarray(point) for point in self.positionSaver[start_idx:end_idx + 1]]

    def evaluateTriangleShape(self) -> tuple[float, float, float]:
        if self.corner1 is None or self.corner2 is None or self.corner1_idx is None or self.corner2_idx is None:
            return 0.0, 0.0, 0.0

        seg1 = self.getSegmentPoints(0, self.corner1_idx)
        seg2 = self.getSegmentPoints(self.corner1_idx, self.corner2_idx)
        seg3 = self.getSegmentPoints(self.corner2_idx, len(self.positionSaver) - 1)

        if len(seg1) < 2 or len(seg2) < 2 or len(seg3) < 2:
            return 0.0, 0.0, 0.0

        dev1 = self.segmentMeanDeviation(self.startPos, self.corner1, seg1)
        dev2 = self.segmentMeanDeviation(self.corner1, self.corner2, seg2)
        # The third edge should be the closing segment back to the start.
        dev3 = self.segmentMeanDeviation(self.corner2, self.startPos, seg3)
        mean_deviation = float(np.mean([dev1, dev2, dev3]))

        area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
        straightness_score = max(0.0, 1.0 - mean_deviation / self.maxMeanLineDeviation)
        return float(area), mean_deviation, float(straightness_score)

    def getEffectiveClosureRadius(self) -> float:
        if self.corner1 is None or self.corner2 is None:
            return float(self.closureRadius)

        edge1 = np.linalg.norm(self.corner1 - self.startPos)
        edge2 = np.linalg.norm(self.corner2 - self.corner1)
        edge3 = np.linalg.norm(self.startPos - self.corner2)
        mean_edge_length = float(np.mean([edge1, edge2, edge3]))

        dynamic_radius = self.closureRadiusRatio * mean_edge_length
        return float(np.clip(dynamic_radius, self.closureRadiusMin, self.closureRadiusMax))

    def getReturnLineDeviation(self, position: Optional[np.ndarray] = None) -> float:
        if self.corner2 is None:
            return 0.0
        position = self.currPos if position is None else position
        return self.pointLineDistance(position, self.corner2, self.startPos)

    def getReturnDirectionAlignment(self, prev_pos: np.ndarray, curr_pos: np.ndarray) -> float:
        if self.corner2 is None:
            return 0.0
        move_dir = self.normalizeVector(curr_pos - prev_pos)
        target_dir = self.normalizeVector(self.startPos - self.corner2)
        return float(np.dot(move_dir, target_dir))

    def updateBestFormSnapshot(self) -> None:
        if self.currPhase != 2 or self.corner1 is None or self.corner2 is None:
            return

        area, mean_deviation, straightness_score = self.evaluateTriangleShape()
        distance_to_start = float(np.linalg.norm(self.currPos - self.startPos))
        effective_closure_radius = self.getEffectiveClosureRadius()
        score = (
            135.0 * straightness_score
            + 20.0 * area
            - 18.0 * self.extraCornerCount
            - 34.0 * distance_to_start
        )
        if distance_to_start <= 1.25 * effective_closure_radius:
            score += 18.0
        if score <= self.bestFormScore:
            return

        self.bestFormScore = float(score)
        self.bestFormSnapshot = {
            "positions": np.array(self.positionSaver, dtype=np.float32).copy(),
            "area": float(area),
            "mean_line_deviation": float(mean_deviation),
            "triangle_straightness": float(straightness_score),
            "distance_to_start": float(distance_to_start),
            "extra_corners": int(self.extraCornerCount),
            "corner1": None if self.corner1 is None else np.array(self.corner1, dtype=np.float32).copy(),
            "corner2": None if self.corner2 is None else np.array(self.corner2, dtype=np.float32).copy(),
            "curriculum_stage": int(self.curriculum_stage),
            "effective_closure_radius": float(effective_closure_radius),
            "success_like": bool(
                distance_to_start <= effective_closure_radius
                and area >= self.minArea
                and self.step_ctr > self.minSteps
            ),
        }

    def getCountedExtraCorners(self) -> int:
        if self.curriculum_stage == 2:
            return min(self.extraCornerCount, self.stage2_maxCountedExtraCorners)
        return int(self.extraCornerCount)

    def capStage2Penalty(self, penalty: float, cap: float) -> float:
        if self.curriculum_stage == 2:
            return float(min(penalty, cap))
        return float(penalty)

    def capLateStagePenalty(self, penalty: float, stage2_cap: float, stage3_cap: float) -> float:
        if self.curriculum_stage == 2:
            return float(min(penalty, stage2_cap))
        if self.curriculum_stage >= 3:
            return float(min(penalty, stage3_cap))
        return float(penalty)
        
    
    def isFinished(self) -> bool:
        #gibt True zurück wenn der aktuelle Zustand auf maximal einen Zentimeter an den Startzustand herankommt, die Mindestanzahl Schritte ausgeführt wurde und der Algorithmus sich in Phase 3 befindet
        distance = math.sqrt((self.currPos[0] - self.startPos[0])**2 + (self.currPos[1] - self.startPos[1])**2)
        area = 0.0
        effective_closure_radius = float(self.closureRadius)
        if self.currPhase == 2:
            area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
            effective_closure_radius = self.getEffectiveClosureRadius()
        if (
            distance <= effective_closure_radius
            and self.step_ctr > self.minSteps
            and self.currPhase == 2
            and area >= self.minArea
        ):
            return True
        return False
    
    def isTruncated(self) ->bool:
        #gibt True zurück wenn die maximale Anzahl an Schritten überschritten wurde
        return self.step_ctr >= self.maxSteps and self.useMaxSteps

    #-------------------
    #---Gymnasium API---
    #-------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        #Variablen zurücksetzen
        self.currPhase = 0
        self.step_ctr = 0
        self.totalReward = 0.0
        self.positionSaver = []
        self.corner1 = None
        self.corner2 = None
        self.corner1_idx = None
        self.corner2_idx = None
        self.lastCornerSteps = 0
        self.extraCornerCount = 0
        self.phase2_target_direction = None
        self.bestFormSnapshot = None
        self.bestFormScore = float("-inf")
        
        angleValues = np.arange(-50, 51)
        self.joint_angles = self.rng.choice(angleValues, size=(3,))
        
        #Positionen und History initialisieren
        self.currPos = self.calculate_new_Position(self.joint_angles)
        
        self.startPos = self.currPos.copy()
        self.positionSaver.append(self.startPos)
        
        #Observation & Info zurückgeben
        obs = self.get_obs()
        info = {"area": 0.0, "terminated": False, "truncated": False,}
        
        return obs, info
    
    def step(self, action):
        '''
        Vorgehen pro Step: 
        - Action: Abweichung je Gelenk um -0.1, +0.1 oder 0 -> übergeben von Agent
        - Beenden der Sequenz wenn Mindestanzahl Schritte gelaufen wurde und Finger sich wieder in Range von 1 Distanzeinheit um den Startpunkt befindet
        - Außerdem Beenden der Sequenz wenn Maximalanzahl Schritte gelaufen wurde
        - 
        '''
        reward = 0.0
        shaping_reward = 0.0
        terminal_reward = 0.0
        terminal_area_bonus = 0.0
        terminal_good_bonus = 0.0
        isPhaseSwitch = False
        wasClipped = False
        terminated = False
        truncated = False
        #Step-Counter erhöhen
        self.step_ctr += 1
        self.lastCornerSteps += 1
        
        #Action auswerten für Protagonist und Antagonist
        if self.useAntagonist:
            actionPro = action["protagonist"]
            actionAnt = action["antagonist"]
        else:
            actionPro = action
            actionAnt = 0
            
        #Gelenkwinkel updaten
        self.updateAngles(actionPro)
        self.updateAngles(actionAnt)
        
        #Prüfen, ob ein Gelenk überdreht
        if np.any(self.joint_angles > self.degree_limit) or np.any(self.joint_angles < -self.degree_limit):
            self.joint_angles = np.clip(self.joint_angles, -self.degree_limit, self.degree_limit)
            wasClipped = True
            
        #neue Position berechnen
        self.currPos = self.calculate_new_Position(self.joint_angles)
        self.positionSaver.append(self.currPos.copy())
        self.maybeInitializePhaseTargetDirection()
        
        #TO-DO: Schauen, ob die Phase sich geändert hat
        if self.isCorner() and self.lastCornerSteps >= self.minCornerSteps and self.currPhase < 2:
            if self.currPhase == 0:
                self.corner1 = self.currPos.copy()
                self.corner1_idx = len(self.positionSaver) - 1
            elif self.currPhase == 1:
                self.corner2 = self.currPos.copy()
                self.corner2_idx = len(self.positionSaver) - 1
            
            isPhaseSwitch = True
            self.lastCornerSteps = 0
            if self.currPhase == 0:
                edge1Len = np.linalg.norm(self.corner1 - self.startPos)
                shaping_reward += self.reward_Corner1 + self.reward_Edge1Len * edge1Len
            if self.currPhase == 1:
                shaping_reward += self.reward_Corner2 + self.reward_Multiplier_Area_Corner2 * self.calculateTriangleArea(self.startPos, self.corner1, self.currPos)
        elif self.currPhase == 2 and self.isCorner() and self.lastCornerSteps >= self.minCornerSteps:
            self.extraCornerCount += 1
            self.lastCornerSteps = 0
            shaping_reward -= self.penalty_ExtraCorner * (
                0.5 if self.curriculum_stage == 2 else 1.0
            )
        
        #Reward-Funktions-block
        if wasClipped:
            shaping_reward -= self.penalty_limitViolation
            
        prevPos = self.positionSaver[-2]
        
        #Rewards für korrekte Bewegungen innerhalb der Phasen        
        match self.currPhase:
            case 0:
                # Subgoal 0: build the first edge by moving outward cleanly.
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                shaping_reward += self.phaseProgressScale0 * (currDistanceToStart - prevDistanceToStart)
                
                if len(self.positionSaver) >= 3 and not isPhaseSwitch:   
                    prevPrevPos = self.positionSaver[-3]
                    turnAngle = self.calculateTurnAngle(prevPrevPos, prevPos, self.currPos)
                    if  turnAngle > self.directionChangeLimit:
                        shaping_reward -= self.penalty_Multiplier_DirectionChange0 * (turnAngle - self.directionChangeLimit)

                if self.step_ctr > self.phase0_soft_limit:
                    shaping_reward -= self.penalty_phaseStall0 * (self.step_ctr - self.phase0_soft_limit)

                segment_points = self.getSegmentPoints(0, len(self.positionSaver) - 1)
                deviation = self.segmentMeanDeviation(self.startPos, self.currPos, segment_points)
                shaping_reward -= self.penalty_SegmentCurvature * deviation

            case 1:
                # Subgoal 1: place the second edge so the triangle spans a large area.
                prevDistanceToCorner1 = np.linalg.norm(prevPos - self.corner1)
                currDistanceToCorner1 = np.linalg.norm(self.currPos - self.corner1)
                shaping_reward += self.phaseProgressScale1 * (currDistanceToCorner1 - prevDistanceToCorner1)
                
                prev_area = self.calculateTriangleArea(self.startPos, self.corner1, prevPos)
                curr_area = self.calculateTriangleArea(self.startPos, self.corner1, self.currPos)
                shaping_reward += self.reward_Multiplier_Area_InSequence * (curr_area - prev_area)

                prev_spread = min(
                    np.linalg.norm(prevPos - self.startPos),
                    np.linalg.norm(prevPos - self.corner1),
                )
                curr_spread = min(
                    np.linalg.norm(self.currPos - self.startPos),
                    np.linalg.norm(self.currPos - self.corner1),
                )
                shaping_reward += self.reward_Phase1Corner2Spread * (curr_spread - prev_spread)
                
                if len(self.positionSaver) >= 3 and not isPhaseSwitch:   
                    prevPrevPos = self.positionSaver[-3]
                    turnAngle = self.calculateTurnAngle(prevPrevPos, prevPos, self.currPos)
                    if  turnAngle > self.directionChangeLimit:
                        shaping_reward -= self.penalty_Multiplier_DirectionChange1 * (turnAngle - self.directionChangeLimit)

                if self.step_ctr > self.phase1_soft_limit:
                    shaping_reward -= self.penalty_phaseStall1 * (self.step_ctr - self.phase1_soft_limit)

                segment_start_idx = self.corner1_idx if self.corner1_idx is not None else 0
                segment_points = self.getSegmentPoints(segment_start_idx, len(self.positionSaver) - 1)
                deviation = self.segmentMeanDeviation(self.corner1, self.currPos, segment_points)
                shaping_reward -= self.penalty_SegmentCurvature * deviation

            case 2:
                # Subgoal 2: return to the start along a clean closing edge.
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                deltaToStart = prevDistanceToStart - currDistanceToStart
                shaping_reward += self.phaseProgressScale2 * deltaToStart
                
                if deltaToStart < 0:
                    away_penalty = self.penalty_Phase2AwayFromStart * abs(deltaToStart)
                    shaping_reward -= self.capLateStagePenalty(
                        away_penalty,
                        self.stage2_awayFromStartPenaltyCap,
                        self.stage3_awayFromStartPenaltyCap,
                    )
                else:
                    shaping_reward += self.reward_Phase2Closure * max(0.0, deltaToStart)

                curr_final_area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
                if self.curriculum_stage < 3:
                    shaping_reward += 0.2 * curr_final_area
                shaping_reward += self.reward_Phase2AreaPreservation * curr_final_area

                alignment = self.getReturnDirectionAlignment(prevPos, self.currPos)
                shaping_reward += self.reward_Phase2DirectionAlignment * alignment

                prev_return_deviation = self.getReturnLineDeviation(prevPos)
                curr_return_deviation = self.getReturnLineDeviation(self.currPos)
                shaping_reward += self.penalty_Phase2ReturnLineDeviation * (prev_return_deviation - curr_return_deviation)
                close_ratio = max(0.0, 1.0 - currDistanceToStart / max(1e-6, self.getEffectiveClosureRadius()))
                line_ratio = max(0.0, 1.0 - curr_return_deviation / max(1e-6, self.maxMeanLineDeviation))
                if close_ratio > 0.0 and line_ratio > 0.0:
                    shaping_reward += self.reward_Phase2CleanReturn * close_ratio * line_ratio

                segment_start_idx = self.corner2_idx if self.corner2_idx is not None else 0
                segment_points = self.getSegmentPoints(segment_start_idx, len(self.positionSaver) - 1)
                deviation = self.segmentMeanDeviation(self.corner2, self.startPos, segment_points)
                curvature_penalty = self.penalty_SegmentCurvature * deviation
                shaping_reward -= self.capLateStagePenalty(
                    curvature_penalty,
                    self.stage2_curvaturePenaltyCap,
                    self.stage3_curvaturePenaltyCap,
                )

                late_phase_window = max(1, self.maxSteps - self.minSteps)
                late_progress = max(0.0, (self.step_ctr - self.minSteps) / late_phase_window)
                late_distance_penalty = self.penalty_Phase2LateDistance * late_progress * currDistanceToStart
                shaping_reward -= self.capLateStagePenalty(
                    late_distance_penalty,
                    self.stage2_lateDistancePenaltyCap,
                    self.stage3_lateDistancePenaltyCap,
                )

                
            case _:
                print("Wir befinden uns in einer ungültigen Phase.")
        
        #optionale rewards für closure oder truncation
        if self.isFinished():
            terminated = True
            final_area, mean_deviation, straightness_score = self.evaluateTriangleShape()
            edge1, edge2, edge3 = self.getTriangleEdgeLengths()
            mean_edge_length, edge_balance = self.getTriangleEdgeMetrics()
            distance_to_start = float(np.linalg.norm(self.currPos - self.startPos))
            terminal_area_bonus = (
                1.35 * final_area * self.reward_Multiplier_Area
                + 1.15 * self.reward_TriangleShape * final_area
            )
            terminal_reward = (
                self.reward_Closure
                + terminal_area_bonus
                + self.reward_TriangleStraightness * straightness_score
                + self.reward_TriangleMeanEdge * mean_edge_length
                + self.reward_TriangleEdgeBalance * edge_balance
                + self.reward_TerminalEdge3Length * edge3
                - self.penalty_ExtraCorner * self.getCountedExtraCorners()
                - self.penalty_TerminalGap * distance_to_start
            )
        elif self.isTruncated():
            #print("Die Sequenz wurde abgeschlossen, war allerdings nicht erfolgreich.")
            truncated = True
            
            if self.corner1 is not None and self.corner2 is not None:
                partial_Area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
                shaping_reward += self.reward_partialArea * partial_Area
                end_distance_penalty = self.penalty_Phase2EndDistance * float(np.linalg.norm(self.currPos - self.startPos))
                shaping_reward -= self.capLateStagePenalty(
                    end_distance_penalty,
                    self.stage2_truncationEndDistancePenaltyCap,
                    self.stage3_truncationEndDistancePenaltyCap,
                )
                _, mean_deviation, straightness_score = self.evaluateTriangleShape()
                shaping_reward += 0.2 * self.reward_TriangleStraightness * straightness_score
                shaping_reward -= 0.5 * self.penalty_ExtraCorner * self.getCountedExtraCorners()
            else:
                shaping_reward -= self.penalty_noTriangle
        
        #Penalty für Schrittanzahl
        shaping_reward -= self.penalty_stepNumberMultiplicator
        reward = shaping_reward + terminal_reward + terminal_good_bonus
        
        #totalReward speichern
        self.totalReward += reward
        
        if isPhaseSwitch:
            self.currPhase += 1
            if self.currPhase == 2:
                self.phase2_target_direction = self.normalizeVector(self.startPos - self.corner2) if self.corner2 is not None else None

        self.updateBestFormSnapshot()
        reward = shaping_reward + terminal_reward + terminal_good_bonus
        #Observation und Info speichern
        obs = self.get_obs()
        current_area = 0.0
        mean_deviation = 0.0
        straightness_score = 0.0
        if self.corner1 is not None and self.corner2 is not None:
            current_area, mean_deviation, straightness_score = self.evaluateTriangleShape()

        distance_to_start = float(np.linalg.norm(self.currPos - self.startPos))
        effective_closure_radius = self.getEffectiveClosureRadius()
        return_line_deviation = float(self.getReturnLineDeviation())
        mean_edge_length, edge_balance = self.getTriangleEdgeMetrics()
        info = {
            "current_Phase": int(self.currPhase),
            "reward": float(reward),
            "shaping_reward": float(shaping_reward),
            "terminal_reward": float(terminal_reward),
            "terminal_area_bonus": float(terminal_area_bonus),
            "terminal_good_bonus": float(terminal_good_bonus),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_ctr": self.step_ctr,
            "has_corner1": bool(self.corner1 is not None),
            "has_corner2": bool(self.corner2 is not None),
            "area": float(current_area),
            "mean_line_deviation": float(mean_deviation),
            "triangle_straightness": float(straightness_score),
            "extra_corners": int(self.extraCornerCount),
            "distance_to_start": distance_to_start,
            "return_line_deviation": return_line_deviation,
            "effective_closure_radius": float(effective_closure_radius),
            "min_straightness_for_success": float(self.minStraightnessForSuccess),
            "max_extra_corners_for_success": int(self.maxExtraCornersForSuccess),
            "mean_edge_length": float(mean_edge_length),
            "edge_balance": float(edge_balance),
            "min_mean_edge_length_for_good_triangle": float(self.minMeanEdgeLengthForGoodTriangle),
            "min_edge_balance_for_good_triangle": float(self.minEdgeBalanceForGoodTriangle),
            "good_triangle": bool(self.isGoodTriangle()),
        }
        
        return obs, reward, terminated, truncated, info
        
        
        
        
