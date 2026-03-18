#Unterstützung von KI in Design-Entscheidungen und teilweise bei Codeteilen

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple, Optional
import math

from env_helpers import (
    calculate_turn_angle,
    calculate_new_position,
    calculate_triangle_area,
    cap_late_stage_penalty,
    evaluate_triangle_shape,
    get_counted_extra_corners,
    get_current_move_direction,
    get_effective_closure_radius,
    get_phase_line_deviation,
    get_phase_progress,
    get_phase_target_direction,
    get_return_direction_alignment,
    get_return_line_deviation,
    get_segment_points,
    get_triangle_edge_lengths,
    get_triangle_edge_metrics,
    is_corner,
    is_good_triangle,
    maybe_initialize_phase_target_direction,
    normalize_vector,
    segment_mean_deviation,
    update_best_form_snapshot,
)

class FingerTriangleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, givenUseAntagonist=False):
        '''
        - Initialisierung von Parametern
        - Definition von Actionspace
        - Definition von Observationspace
        '''
        
        super().__init__()
        
        #einstellbare Parameter
        self.useMaxSteps = True
        self.useAntagonist = bool(givenUseAntagonist)
        self.maxSteps = 150
        self.maxStepsStage0 = self.maxSteps * 0.93
        self.maxStepsStage1 = self.maxSteps * 1.01
        self.maxStepsStage2 = self.maxSteps * 1.06
        self.maxStepsStage3 = self.maxSteps * 1.1
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
        self.phase0_soft_limit = 40
        self.phase1_soft_limit = 90
        self.maxMeanLineDeviation = 0.35
        self.minStraightnessForSuccess = 0.35
        self.maxExtraCornersForSuccess = 2
        self.minMeanEdgeLengthForGoodTriangle = 1.0
        self.minEdgeBalanceForGoodTriangle = 0.45
        self.curriculum_stage = 0
        
        #penalties and rewards
        self.reward_Closure = 150.0
        self.reward_Corner1 = 25.0
        self.reward_Corner2 = 40.0
        self.reward_Multiplier_Area_Corner2 = 60.0
        self.reward_Multiplier_Area = 75.0
        self.reward_Multiplier_Area_InSequence = 75.0
        self.reward_partialArea = 12.0
        self.reward_Edge1Len = 0.8
        self.reward_Phase2Closure = 10.0
        self.reward_Phase2AreaPreservation = 0.0
        self.reward_TriangleShape = 138.0
        self.reward_TriangleStraightness = 110.0
        self.reward_Phase1Corner2Spread = 9.0
        self.reward_Phase1Corner1Separation = 2.5
        self.reward_TriangleMeanEdge = 20.0
        self.reward_TriangleEdgeBalance = 28.0
        self.reward_Phase2DirectionAlignment = 2.0
        self.reward_Phase2CleanReturn = 0.0
        self.penalty_limitViolation = 0.05
        self.penalty_noTriangle = 15.0
        self.penalty_stepNumberMultiplicator = 0.015
        self.penalty_Multiplier_DirectionChange0 = 0.00
        self.penalty_Multiplier_DirectionChange1 = 0.0008
        self.penalty_Phase1PrematureReturn = 2.0
        self.penalty_Phase2AwayFromStart = 8.0
        self.penalty_Phase2EndDistance = 4.0
        self.penalty_ExtraCorner = 16.0
        self.penalty_SegmentCurvature = 5.0
        self.penalty_phaseStall0 = 0.04
        self.penalty_phaseStall1 = 0.07
        self.penalty_Phase2ReturnLineDeviation = 6.0
        self.penalty_Phase2LateDistance = 1.5
        self.phaseProgressScale0 = 0.05
        self.phaseProgressScale1 = 0.8
        self.phaseProgressScale2 = 10.0
        self.stage2_maxCountedExtraCorners = 2
        self.stage2_curvaturePenaltyCap = 10.0
        self.stage2_awayFromStartPenaltyCap = 10.0
        self.stage2_lateDistancePenaltyCap = 8.0
        self.stage2_truncationEndDistancePenaltyCap = 8.0
        self.stage3_curvaturePenaltyCap = 8.0
        self.stage3_awayFromStartPenaltyCap = 8.0
        self.stage3_lateDistancePenaltyCap = 6.0
        self.stage3_truncationEndDistancePenaltyCap = 6.0
        
        self.penalty_TerminalGap = 10.0
        self.reward_TerminalEdge3Length = 6.0
        self.partialAreaScale = 1.0
        self.terminalGapScale = 1.0
    
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
        #Diskret mit 27 Aktionen, Aktionsdefinition liegt in "Actions.txt"
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
      
    
    
    #---Hilfsfunktionen------------------------------------------------------------------------------
    
    
    def get_obs(self) -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        
        #Relative Position zum Start
        distancetoStart = (self.currPos - self.startPos).astype(np.float32)

        # Relative Position zu corner1
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
            desiredReturnDirection = normalize_vector(self.startPos - self.corner2).astype(np.float32)

        #Observation-Metriken berechnen
        currentMoveDirection = get_current_move_direction(self).astype(np.float32)
        effectiveClosureRadius = np.array([get_effective_closure_radius(self)], dtype=np.float32)
        returnLineDeviation = np.array([get_return_line_deviation(self)], dtype=np.float32)
        distanceToStartNorm = np.array([np.linalg.norm(self.currPos - self.startPos)], dtype=np.float32)
        phaseTargetDirection = get_phase_target_direction(self).astype(np.float32)
        phaseLineDeviation = np.array([get_phase_line_deviation(self)], dtype=np.float32)
        phaseProgress = np.array([get_phase_progress(self)], dtype=np.float32)
        
        #Phase für Observation als one hot Array definieren
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
        #Setzt Werte für Variablen, die sich über die Stages verändern, je nachdem, welche Stage aktuell ist
        self.curriculum_stage = int(stage)

        if self.curriculum_stage <= 0:
            #Stage 0: lockere Bedingungen
            self.maxSteps = self.maxStepsStage0
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
            self.reward_Phase1Corner2Spread = 8.0
            self.reward_Phase1Corner1Separation = 2.0
            self.penalty_Phase1PrematureReturn = 1.5
        elif self.curriculum_stage == 1:
            #Stage 1: Verschärfung von Parametern
            self.maxSteps = self.maxStepsStage1
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
            self.reward_Phase1Corner2Spread = 9.0
            self.reward_Phase1Corner1Separation = 2.5
            self.penalty_Phase1PrematureReturn = 2.0
            self.partialAreaScale = 1.0
            self.terminalGapScale = 1.0
        elif self.curriculum_stage == 2:
            #Stage 2: weitere Verschärfung
            self.maxSteps = self.maxStepsStage2
            self.minSteps = 22
            self.closureRadius = 1.72
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
            self.reward_Phase1Corner2Spread = 7.5
            self.reward_Phase1Corner1Separation = 2.0
            self.penalty_Phase1PrematureReturn = 1.5
            self.penalty_Phase2AwayFromStart = 8.0
            self.penalty_Phase2ReturnLineDeviation = 6.0
            self.penalty_Phase2LateDistance = 1.5
            self.partialAreaScale = 1.0
            self.terminalGapScale = 1.0
        else:
            #Stage 3: nahezu identisch zu Stage 2, nur als kurze Abschlussphase
            self.maxSteps = self.maxStepsStage3
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
            self.reward_Phase2Closure = 11.3
            self.reward_Phase2AreaPreservation = 0.0
            self.reward_Phase2DirectionAlignment = 2.4
            self.reward_Phase2CleanReturn = 1.4
            self.reward_Phase1Corner2Spread = 4.5
            self.reward_Phase1Corner1Separation = 1.0
            self.penalty_Phase1PrematureReturn = 0.6
            self.penalty_Phase2AwayFromStart = 8.0
            self.penalty_Phase2ReturnLineDeviation = 5.8
            self.penalty_Phase2LateDistance = 1.2
            self.partialAreaScale = 1.0
            self.terminalGapScale = 1.0
    
    def updateAngles(self, action, antagonist): 
        #aktualisiert die Gelenkwinkel auf Basis der übergebenen Aktion
        
        if antagonist:
            self.action_delta = 0.2
        else:
            self.action_delta = 1.0
        #Gelenk 1 aktualisieren 
        if action > 8 and action <= 17:
            self.joint_angles[0] -= self.action_delta
        elif action > 17:
            self.joint_angles[0] += self.action_delta
        
        #Gelenk 2 aktualisieren
        if (action > 2 and action < 6) or (action > 11 and action < 15) or (action > 20 and action < 24):
            self.joint_angles[1] -= self.action_delta
        elif (action > 5 and action < 9) or (action > 14 and action < 18) or action > 23:
            self.joint_angles[1] += self.action_delta
        
        #Gelenk 3 aktualisieren
        if (action)%3 == 1:
            self.joint_angles[2] -= self.action_delta
        elif (action)%3 == 2:
            self.joint_angles[2] += self.action_delta
            
    def isFinished(self) -> bool:
        #gibt True zurück wenn der aktuelle Zustand in den effektiven closure-Radios um den Startzustand herankommt, die Mindestanzahl Schritte ausgeführt wurde, eine Mindestfläche erreich wurde und der Algorithmus sich in Phase 2 befindet
        distance = math.sqrt((self.currPos[0] - self.startPos[0])**2 + (self.currPos[1] - self.startPos[1])**2)
        area = 0.0
        effective_closure_radius = float(self.closureRadius)
        
        if self.currPhase == 2:
            area = calculate_triangle_area(self.startPos, self.corner1, self.corner2)
            effective_closure_radius = get_effective_closure_radius(self)
        
        if (distance <= effective_closure_radius and self.step_ctr > self.minSteps and self.currPhase == 2 and area >= self.minArea):
            return True
        return False
    
    def isTruncated(self) ->bool:
        #gibt True zurück wenn die maximale Anzahl an Schritten überschritten wurde
        return self.step_ctr >= self.maxSteps and self.useMaxSteps



    #---Gymnasium API------------------------------------------------------------------------------


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        #Umgebung zurücksetzen
        
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        #Variablen zurücksetzen
        self.currPhase = 0
        self.step_ctr = 0
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
        
        #Startwinkel der Gelenke zufällig zwischen -50 bis 50 Grad
        angleValues = np.arange(-50, 51)
        self.joint_angles = self.rng.choice(angleValues, size=(3,))
        
        #Positionen und History initialisieren
        self.currPos = calculate_new_position(self.joint_angles, self.link_lengths)
        
        self.startPos = self.currPos.copy()
        self.positionSaver.append(self.startPos)
        
        #Observation & Info zurückgeben
        obs = self.get_obs()
        info = {"area": 0.0, "terminated": False, "truncated": False,}
        
        return obs, info
    
    def step(self, action):
        '''
        Vorgehen pro Step: 
        - Action Protagonist: Abweichung je Gelenk um -1, +1 oder 0 -> übergeben von Agent
        - Action Antagonist: Abweichung je Gelenk um -0.2, +0.2 oder 0 -> übergeben von Antagonist
        - Prüfen, ob ein Gelenk überdreht
        - Neue Position berechnen
        - Prüfen, ob Ecke gefunden wurde und Phase sich ändert
        - Reward des Schritts auf Basis der Phase errechnen
        - Beenden der Sequenz wenn Mindestanzahl Schritte gelaufen wurde und Finger sich wieder in definiertem Radius um den Startpunkt befindet
        - Außerdem Beenden der Sequenz wenn Maximalanzahl Schritte gelaufen wurde
        - 
        '''
        reward = 0.0
        shaping_reward = 0.0
        terminal_reward = 0.0
        terminal_area_bonus = 0.0
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
        self.updateAngles(actionPro, False)
        self.updateAngles(actionAnt, True)
        
        #Prüfen, ob ein Gelenk überdreht
        if np.any(self.joint_angles > self.degree_limit) or np.any(self.joint_angles < -self.degree_limit):
            self.joint_angles = np.clip(self.joint_angles, -self.degree_limit, self.degree_limit)
            wasClipped = True
            
        #neue Position berechnen
        self.currPos = calculate_new_position(self.joint_angles, self.link_lengths)
        self.positionSaver.append(self.currPos.copy())
        maybe_initialize_phase_target_direction(self)
        
        #Schauen, ob die Phase sich geändert hat
        if is_corner(self) and self.lastCornerSteps >= self.minCornerSteps and self.currPhase < 2:
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
                shaping_reward += self.reward_Corner2 + self.reward_Multiplier_Area_Corner2 * calculate_triangle_area(self.startPos, self.corner1, self.currPos)
        elif self.currPhase == 2 and is_corner(self) and self.lastCornerSteps >= self.minCornerSteps:
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
                # Subziel 0: Gerade Kante vom Start weg aufbauen und erste Ecke bilden
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                shaping_reward += self.phaseProgressScale0 * (currDistanceToStart - prevDistanceToStart)
                
                #Überziehen der Phase
                if self.step_ctr > self.phase0_soft_limit:
                    shaping_reward -= self.penalty_phaseStall0 * (self.step_ctr - self.phase0_soft_limit)

                segment_points = get_segment_points(self, 0, len(self.positionSaver) - 1)
                deviation = segment_mean_deviation(self.startPos, self.currPos, segment_points)
                shaping_reward -= self.penalty_SegmentCurvature * deviation

            case 1:
                # Subziel 1: Gerade Kante die von Ecke 1 wegführt und eine große Fläche aufspannt
                
                #Distanzerhöhung
                prevDistanceToCorner1 = np.linalg.norm(prevPos - self.corner1)
                currDistanceToCorner1 = np.linalg.norm(self.currPos - self.corner1)
                shaping_reward += self.phaseProgressScale1 * (currDistanceToCorner1 - prevDistanceToCorner1)
                shaping_reward += self.reward_Phase1Corner1Separation * (currDistanceToCorner1 - prevDistanceToCorner1)
                
                #Flächenerhöhung
                prev_area = calculate_triangle_area(self.startPos, self.corner1, prevPos)
                curr_area = calculate_triangle_area(self.startPos, self.corner1, self.currPos)
                shaping_reward += self.reward_Multiplier_Area_InSequence * (curr_area - prev_area)

                #Erhöhung der Aufspannung
                prev_spread = min(
                    np.linalg.norm(prevPos - self.startPos),
                    np.linalg.norm(prevPos - self.corner1),
                )
                curr_spread = min(
                    np.linalg.norm(self.currPos - self.startPos),
                    np.linalg.norm(self.currPos - self.corner1),
                )
                shaping_reward += self.reward_Phase1Corner2Spread * (curr_spread - prev_spread)

                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                if currDistanceToStart < prevDistanceToStart:
                    shaping_reward -= self.penalty_Phase1PrematureReturn * (prevDistanceToStart - currDistanceToStart)
                
                #Überziehen der Phase
                if self.step_ctr > self.phase1_soft_limit:
                    shaping_reward -= self.penalty_phaseStall1 * (self.step_ctr - self.phase1_soft_limit)

                segment_start_idx = self.corner1_idx if self.corner1_idx is not None else 0
                
                #Abweichung von der Phasenlinie
                segment_points = get_segment_points(self, segment_start_idx, len(self.positionSaver) - 1)
                deviation = segment_mean_deviation(self.corner1, self.currPos, segment_points)
                shaping_reward -= self.penalty_SegmentCurvature * deviation

            case 2:
                # Subziel 2: gerade Kante zurück zum Start
                
                #Distanzverkürzung
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                deltaToStart = prevDistanceToStart - currDistanceToStart
                #shaping_reward += self.phaseProgressScale2 * deltaToStart
                
                #bei starker Nähe zum Startpunkt gibt es nur einen maximalen Penaltywert
                if deltaToStart < 0:
                    away_penalty = self.penalty_Phase2AwayFromStart * abs(deltaToStart)
                    shaping_reward -= cap_late_stage_penalty(self, away_penalty, self.stage2_awayFromStartPenaltyCap, self.stage3_awayFromStartPenaltyCap)
                else:
                    shaping_reward += self.reward_Phase2Closure * max(0.0, deltaToStart)
                
                #zu später Zeit der Phase extra penalty für Distanz zum Ziel
                late_phase_window = max(1, self.maxSteps - self.minSteps)
                late_progress = max(0.0, (self.step_ctr - self.minSteps) / late_phase_window)
                late_distance_penalty = self.penalty_Phase2LateDistance * late_progress * currDistanceToStart
                shaping_reward -= cap_late_stage_penalty(self, late_distance_penalty, self.stage2_lateDistancePenaltyCap, self.stage3_lateDistancePenaltyCap)

                #Abweichungsänderung von Phasenlinie
                prev_return_deviation = get_return_line_deviation(self, prevPos)
                curr_return_deviation = get_return_line_deviation(self, self.currPos)
                shaping_reward += self.penalty_Phase2ReturnLineDeviation * (prev_return_deviation - curr_return_deviation)
                
                close_ratio = max(0.0, 1.0 - currDistanceToStart / max(1e-6, get_effective_closure_radius(self)))
                line_ratio = max(0.0, 1.0 - curr_return_deviation / max(1e-6, self.maxMeanLineDeviation))
                if close_ratio > 0.0 and line_ratio > 0.0:
                    shaping_reward += self.reward_Phase2CleanReturn * close_ratio * line_ratio

                
                
            case _:
                print("Wir befinden uns in einer ungültigen Phase.")
        
        #optionale rewards für closure oder truncation
        if self.isFinished():
            #terminal-Metrics berechnen
            terminated = True
            final_area, mean_deviation, straightness_score = evaluate_triangle_shape(self)
            edge1, edge2, edge3 = get_triangle_edge_lengths(self)
            mean_edge_length, edge_balance = get_triangle_edge_metrics(self)
            distance_to_start = float(np.linalg.norm(self.currPos - self.startPos))
            terminal_area_bonus = (final_area * self.reward_Multiplier_Area + self.reward_TriangleShape * final_area)
            
            #terminalreward zusammenrechnen
            terminal_reward = (
                self.reward_Closure
                + terminal_area_bonus
                + self.reward_TriangleStraightness * straightness_score
                + self.reward_TriangleMeanEdge * mean_edge_length
                + self.reward_TriangleEdgeBalance * edge_balance
                + self.reward_TerminalEdge3Length * edge3
                - self.penalty_ExtraCorner * get_counted_extra_corners(self)
                - self.penalty_TerminalGap * self.terminalGapScale * distance_to_start
            )
        elif self.isTruncated():
            #Truncation-Metrics berechnen
            truncated = True
            
            if self.corner1 is not None and self.corner2 is not None:
                #Teilreward für richtige Form
                partial_Area = calculate_triangle_area(self.startPos, self.corner1, self.corner2)
                shaping_reward += self.reward_partialArea * self.partialAreaScale * partial_Area
                end_distance_penalty = self.penalty_Phase2EndDistance * float(np.linalg.norm(self.currPos - self.startPos))
                shaping_reward -= cap_late_stage_penalty(self, end_distance_penalty, self.stage2_truncationEndDistancePenaltyCap, self.stage3_truncationEndDistancePenaltyCap)
                _, mean_deviation, straightness_score = evaluate_triangle_shape(self)
                
                shaping_reward += 0.2 * self.reward_TriangleStraightness * straightness_score
                shaping_reward -= 0.5 * self.penalty_ExtraCorner * get_counted_extra_corners(self)
            else:
                shaping_reward -= self.penalty_noTriangle
        
        #Penalty für Schrittanzahl
        shaping_reward -= self.penalty_stepNumberMultiplicator
        reward = shaping_reward + terminal_reward
        
        if isPhaseSwitch:
            self.currPhase += 1
            if self.currPhase == 2:
                self.phase2_target_direction = normalize_vector(self.startPos - self.corner2) if self.corner2 is not None else None

        #Wenn neue beste Form gefunden wurde speichern
        update_best_form_snapshot(self)
        
        #Observation und Info speichern
        obs = self.get_obs()
        current_area = 0.0
        mean_deviation = 0.0
        straightness_score = 0.0
        if self.corner1 is not None and self.corner2 is not None:
            current_area, mean_deviation, straightness_score = evaluate_triangle_shape(self)

        #Info-Werte berechnen für Plotting
        distance_to_start = float(np.linalg.norm(self.currPos - self.startPos))
        effective_closure_radius = get_effective_closure_radius(self)
        return_line_deviation = float(get_return_line_deviation(self))
        mean_edge_length, edge_balance = get_triangle_edge_metrics(self)
        info = {
            "current_Phase": int(self.currPhase),
            "reward": float(reward),
            "shaping_reward": float(shaping_reward),
            "terminal_reward": float(terminal_reward),
            "terminal_area_bonus": float(terminal_area_bonus),
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
            "good_triangle": bool(is_good_triangle(self)),
        }
        
        return obs, reward, terminated, truncated, info
        
        
        
        
