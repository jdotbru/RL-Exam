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
        self.maxSteps = 150
        self.minSteps = 30
        self.angleThresholdMin = 15.0
        self.angleThresholdMax = 150.0
        self.minCornerSteps = 5
        self.min_segment_len = 0.2
        self.action_delta = 1.0
        self.closureRadius = 1.0
        self.cornerWindow = 3
        self.directionChangeLimit = 10.0
        self.minArea = 0.3
        self.targetAngle = 90.0
        
        #penalties and rewards
        self.totalReward = 0
        self.reward_Closure = 1000.0
        self.reward_Corner1 = 10.0
        self.reward_Corner2 = 2.0
        self.reward_Multiplier_Area_Corner2 = 20.0
        self.reward_Multiplier_Area = 60.0
        self.reward_Multiplier_Area_InSequence = 40.0
        self.reward_partialArea = 0.0
        self.reward_Edge1Len = 0.8
        self.penalty_limitViolation = 0.05
        self.penalty_noTriangle = 10.0
        self.penalty_stepNumberMultiplicator = 0.01
        self.penalty_Multiplier_DirectionChange0 = 0.01
        self.penalty_Multiplier_DirectionChange1 = 0.002
        self.penalty_Multiplier_Parallelity = 4.0
        self.penalty_Multiplier_AngleOptimality = 3.0
        self.phaseProgressScale0 = 0.1
        self.phaseProgressScale1 = 1.2
        self.phaseProgressScale2 = 6.0
    
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
        self.lastCornerSteps = 0
        
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
        obs_low = np.array([-self.degree_limit]*3 + [-10]*2 + [-20]*2 +[-20]*2 +[-20]*2 + [0], dtype=np.float32)
        obs_high = np.array([self.degree_limit]*3 + [10]*2 + [20]*2 + [20]*2 +[20]*2 + [2], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    
    #---------------------    
    #---Hilfsfunktionen---
    #---------------------
    def get_obs(self) -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        distancetoStart = (self.currPos - self.startPos).astype(np.float32)
        
        if self.corner1 is None:
            distanceToCorner1 = np.zeros(2, dtype=np.float32)
        else:
            distanceToCorner1 = (self.currPos - self.corner1).astype(np.float32)

        # Relative Position zu corner2
        if self.corner2 is None:
            distanceToCorner2 = np.zeros(2, dtype=np.float32)
        else:
            distanceToCorner2 = (self.currPos - self.corner2).astype(np.float32)
        
        phase = np.array([self.currPhase], dtype=np.float32)
        return np.concatenate([self.joint_angles, self.currPos, distancetoStart, distanceToCorner1, distanceToCorner2, phase]).astype(np.float32)
    
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
        
    
    def isFinished(self) -> bool:
        #gibt True zurück wenn der aktuelle Zustand auf maximal einen Zentimeter an den Startzustand herankommt, die Mindestanzahl Schritte ausgeführt wurde und der Algorithmus sich in Phase 3 befindet
        distance = math.sqrt((self.currPos[0] - self.startPos[0])**2 + (self.currPos[1] - self.startPos[1])**2)
        if self.currPhase == 2:
            area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
        if (distance <= self.closureRadius and self.step_ctr > self.minSteps and self.currPhase == 2 and area >= self.minArea):
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
        self.lastCornerSteps = 0
        
        #Initiale Gelenkwinkel setzen (diskrete Werte im Intervall von 0.1 zwischen -1 und 1)
        angleValues = np.arange(-60, 61)
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
        
        #TO-DO: Schauen, ob die Phase sich geändert hat
        if self.isCorner() and self.lastCornerSteps >= self.minCornerSteps and self.currPhase < 2:
            if self.currPhase == 0:
                self.corner1 = self.currPos.copy()
            elif self.currPhase == 1:
                self.corner2 = self.currPos.copy()
            
            isPhaseSwitch = True
            self.lastCornerSteps = 0
            if self.currPhase == 0:
                edge1Len = np.linalg.norm(self.corner1 - self.startPos)
                reward += self.reward_Corner1 + self.reward_Edge1Len * edge1Len
            if self.currPhase == 1:
                reward += self.reward_Corner2 + self.reward_Multiplier_Area_Corner2 * self.calculateTriangleArea(self.startPos, self.corner1, self.currPos)
        
        #Reward-Funktions-block
        if wasClipped:
            reward -= self.penalty_limitViolation
            
        prevPos = self.positionSaver[-2]
        
        #Rewards für korrekte Bewegungen innerhalb der Phasen        
        match self.currPhase:
            case 0:
                #erste Kante soll vom Start weg und gerade verlaufen
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                reward += self.phaseProgressScale0 * (currDistanceToStart - prevDistanceToStart)
                
                if len(self.positionSaver) >= 3 and not isPhaseSwitch:   
                    prevPrevPos = self.positionSaver[-3]
                    turnAngle = self.calculateTurnAngle(prevPrevPos, prevPos, self.currPos)
                    if  turnAngle > self.directionChangeLimit:
                        reward -= self.penalty_Multiplier_DirectionChange0 * (turnAngle - self.directionChangeLimit)
            case 1:
                #zweite Phase soll eine große Fläche erzeugen
                prevDistanceToCorner1 = np.linalg.norm(prevPos - self.corner1)
                currDistanceToCorner1 = np.linalg.norm(self.currPos - self.corner1)
                #Bewegung von Ecke 1 belohnen
                reward += self.phaseProgressScale1 * (currDistanceToCorner1 - prevDistanceToCorner1)
                
                prev_area = self.calculateTriangleArea(self.startPos, self.corner1, prevPos)
                curr_area = self.calculateTriangleArea(self.startPos, self.corner1, self.currPos)
                #große Fläche belohnen
                reward += self.reward_Multiplier_Area_InSequence * (curr_area - prev_area)
                
                parallelity = self.calculateParallelity()
                #Parallelität zu erster kante bestrafen
                reward -= self.penalty_Multiplier_Parallelity * max(0.0, parallelity)
                
                angleEdge2 = self.calculateAngleOfEdge2()
                #Winkel zwischen erster und zweiter Kante belohnen
                reward += self.penalty_Multiplier_AngleOptimality * (1 - abs(angleEdge2 - self.targetAngle) / self.targetAngle)
                
                if len(self.positionSaver) >= 3 and not isPhaseSwitch:   
                    prevPrevPos = self.positionSaver[-3]
                    turnAngle = self.calculateTurnAngle(prevPrevPos, prevPos, self.currPos)
                    if  turnAngle > self.directionChangeLimit:
                        reward -= self.penalty_Multiplier_DirectionChange1 * (turnAngle - self.directionChangeLimit)
            case 2:
                #dritte Phase soll zurück zum Start kommen und dabei die Fläche erhalten
                prevDistanceToStart = np.linalg.norm(prevPos - self.startPos)
                currDistanceToStart = np.linalg.norm(self.currPos - self.startPos)
                deltaToStart = prevDistanceToStart - currDistanceToStart
                #Annäherung an den Start belohnen
                reward += self.phaseProgressScale2 * deltaToStart
                
                if deltaToStart < 0:
                    reward += self.phaseProgressScale2 * deltaToStart
                             
                #curr_final_area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
                #reward += 0.1 * curr_final_area 
                
            case _:
                print("Wir befinden uns in einer ungültigen Phase.")
        
        #optionale rewards für closure oder truncation
        if self.isFinished():
            #print("Die Sequenz wurde erfolgreich abgeschlossen.")
            terminated = True
            reward += self.reward_Closure + (self.calculateTriangleArea(self.startPos, self.corner1, self.corner2) * self.reward_Multiplier_Area)
        elif self.isTruncated():
            #print("Die Sequenz wurde abgeschlossen, war allerdings nicht erfolgreich.")
            truncated = True
            
            if self.corner1 is not None and self.corner2 is not None:
                partial_Area = self.calculateTriangleArea(self.startPos, self.corner1, self.corner2)
                reward += self.reward_partialArea * partial_Area
            else:
                reward -= self.penalty_noTriangle
        
        #Penalty für Schrittanzahl
        reward -= self.penalty_stepNumberMultiplicator
        
        #totalReward speichern
        self.totalReward += reward
        
        if isPhaseSwitch:
            self.currPhase += 1
        #Observation und Info speichern
        obs = self.get_obs()
        info = {"current_Phase": int(self.currPhase),"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated), "step_ctr": self.step_ctr}
        
        return obs, reward, terminated, truncated, info
        
        
        
        