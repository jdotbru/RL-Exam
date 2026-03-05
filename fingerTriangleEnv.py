from random import seed
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, Tuple, Optional
import math
from itertools import combinations

class FingerTriangleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, givenUseAntagonist=False):
        super().__init__()
        
        #einstellbare Parameter
        self.useMaxSteps = True
        self.useAntagonist = bool(givenUseAntagonist)
        self.maxSteps = 500
        self.minSteps = 50
        self.protagonist_max_deg_per_step = 1.0
        self.antagonist_max_deg_per_step = 0.2
        
        #penalties and rewards
        self.totalReward = 0
        self.reward_Closure = 10.0
        self.penalty_limitViolation = 0.05
        self.penalty_noTriangle = 1.0
        self.penalty_stepNumberMultiplicator = 0.01
    
        #gesetzte Parameter
        self.link_lengths = np.array([5.0, 2.5, 2.5])
        self.degree_limit_rad = math.pi/2
        
        #Random Number Generator & State-Variablen
        self.rng = np.random.default_rng(0)
        self.step_ctr = 0
        self.joint_angles = np.zeros(3)
        self.currPos = np.zeros(2)
        self.startPos = np.zeros(2)
        self.positionSaver: list[np.ndarray] = []
        self.bestAreaSoFar = 0.0
        
        #Action-space
        if self.useAntagonist:
            self.action_space = spaces.Dict({
                "protagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "antagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            })
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        #Observation-space
        #Grenzen des Fingers liegen bei -10:10 (vollausgestreckte addition aller Fingerteile), somit liegen die Grenzen der Differenz zwischen zwei Punkten bei -20:20
        obs_low = np.array([-self.degree_limit_rad]*3 + [-10]*2 + [-20]*2, dtype=np.float32)
        obs_high = np.array([self.degree_limit_rad]*3 + [10]*2 + [20]*2, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    
    #---------------------    
    #---Hilfsfunktionen---
    #---------------------
    def get_obs(self) -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        distancetoStart = (self.currPos - self.startPos).astype(np.float32)
        return np.concatenate([self.joint_angles, self.currPos, distancetoStart]).astype(np.float32)
     
    def calculate_new_Position(self, angles: np.ndarray) -> np.ndarray:
        #berechnet neue Position des Fingers
        p1, p2, p3 = self.link_lengths
        a1, a2, a3 = angles
        
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
    
    def crossProduct(self, origin, a, b) -> float:
        #berechnet ob durch das neue Element b eine Rechtsdrehung der Punkte entsteht => mittleres Element dann innerhalb der Hülle der beiden anderen Punkte
        return (a[0]-origin[0])*(b[1]-origin[1]) - (a[1]-origin[1])*(b[0]-origin[0])
    
    def convex_hull(self, pts:list[np.ndarray]) -> list[np.ndarray]:
        #Bildet die Konvexhülle einer Menge von Punkten, sodass das Kreuzprodukt deutlich schneller abläuft. 
        if len(pts) <= 2:
            return pts
        
        #Punkte sortieren, damit das Bilden der Konvexhülle sinnvoll abgearbeitet werden kann
        pts_sorted = sorted(pts, key=lambda p: (p[0], p[1]))
        
        #untere Hülle
        lower = []
        for p in pts_sorted:
            #poppt Eintrag wenn Rechtsdrehung oder kollinearität vorhanden ist
            while len(lower) >= 2 and self.crossProduct(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        #obere Hülle
        upper = []
        for p in reversed(pts_sorted):
            #poppt Eintrag wenn Rechtsdrehung oder kollinearität vorhanden ist
            while len(upper) >= 2 and self.crossProduct(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
            
        hull = lower[:-1] + upper[:-1]
        return hull
        
    
    def getMaxArea(self) -> float:
        #Identifiziert die drei Eckpunkte des größtmöglichen Dreiecks aus den besuchten punkten des Fingers und berechnet daraus die Fläche.
        
        #Duplikate entfernen durch umwandlung in Liste von Tupeln, da auf arrays kein Set anwendbar ist
        uniquePointtuples = list(set(tuple(np.round(p, 4)) for p in self.positionSaver))
        uniquePositionSaver = list(np.array(p) for p in uniquePointtuples)
        
        #TO-DO Konvexhülle bilden für weniger Punkte...
        cvx_hull = self.convex_hull(uniquePositionSaver)
            
        #Prüfen, ob genug Punkte gesammelt wurden
        points = cvx_hull
        if len(points) < 3:
            return 0.0
        
        #Alle möglichen Kombinationen generieren und Ecken des größtmöglichen Dreiecks finden
        maxArea = 0.0
        
        for p1, p2, p3 in combinations(points, 3):
            area = self.calculateTriangleArea(p1, p2, p3)
            if area > maxArea:
                maxArea = area
        return maxArea
    
    def isFinished(self) -> bool:
        #gibt True zurück wenn der aktuelle Zustand auf maximal einen Zentimeter an den Startzustand herankommt und die Mindestanzahl Schritte ausgeführt wurde
        distance = math.sqrt((self.currPos[0] - self.startPos[0])**2 + (self.currPos[1] - self.startPos[1])**2)
        if (distance <= 1 and self.step_ctr > self.minSteps):
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
        self.step_ctr = 0
        self.bestAreaSoFar = 0.0
        self.totalReward = 0.0
        self.positionSaver = []
        
        #Initiale Gelenkwinkel setzen (zufällig klein um 0)
        self.joint_angles = self.rng.uniform(low=-0.1, high=0.1, size=(3,))
        
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
        - Action: Abweichung je Gelenk zwischen -1 und 1 Grad -> übergeben von Agent
        - Beenden der Sequenz wenn Mindestanzahl Schritte gelaufen wurde und Finger sich wieder in Range von 1 Distanzeinheit um den Startpunkt befindet
        - Außerdem Beenden der Sequenz wenn Maximalanzahl Schritte gelaufen wurde
        - 
        '''
        reward = 0.0
        #Step-Counter erhöhen
        self.step_ctr += 1
        
        #Action auswerten für Protagonist und Antagonist
        if self.useAntagonist:
            actionPro = np.asarray(action["protagonist"])
            actionAnt = np.asarray(action["antagonist"])
        else:
            actionPro = np.asarray(action)
            actionAnt = np.zeros(3)
            
        #Actions clippen, sodass durch keine fehlerhaften Policies eine Drehung von mehr als einem Grad entstehen kann
        actionPro = np.clip(actionPro, -1.0, 1.0)
        actionAnt = np.clip(actionAnt, -1.0, 1.0)
        
        #Action in Delta der Einheit Rad interpretieren
        delta_deg = actionPro * self.protagonist_max_deg_per_step + actionAnt * self.antagonist_max_deg_per_step
        delta = delta_deg * (math.pi / 180.0)
        self.joint_angles += delta
        
        #Prüfen, ob ein Gelenk überdreht
        if np.any(self.joint_angles > self.degree_limit_rad) or np.any(self.joint_angles < -self.degree_limit_rad):
            print("Diese Aktion kann nicht ausgeführt werden, da eines der Gelenke überdrehen würde. Die Gelenke werden an der maximalen Gelenkgrenze gestoppt. ")
            for x in self.joint_angles:
                print(x, ", ")
            self.joint_angles = np.clip(self.joint_angles, -self.degree_limit_rad, self.degree_limit_rad)
            reward -= self.penalty_limitViolation
            
        
        #neue Position berechnen
        self.currPos = self.calculate_new_Position(self.joint_angles)
        self.positionSaver.append(self.currPos.copy())
        
        
        #Prüfen, ob Sequenz zuende ist
        terminated = False
        truncated = False
        
        #optionale rewards für closure oder truncation
        if self.isFinished():
            print("Die Sequenz wurde erfolgreich abgeschlossen.")
            terminated = True
            reward += self.reward_Closure
        elif self.isTruncated():
            print("Die Sequenz wurde abgeschlossen, war allerdings nicht erfolgreich.")
            truncated = True
            #reward -= self.penalty_noTriangle
        
        #reward für größte Fläche nach dem step (positiv wenn größere Fläche vorhanden ist, sonst null)
        currArea = self.getMaxArea()
        deltaArea = currArea - self.bestAreaSoFar
        
        #durch numerische Fehler in der Hull kann die maxArea leicht kleiner werden, daher nur positive Änderungen der maxArea bewerten
        reward += max(0.0, deltaArea)
        self.bestAreaSoFar = max(self.bestAreaSoFar, currArea)
        
        #Penalty für Schrittanzahl
        reward -= self.penalty_stepNumberMultiplicator
        
        #totalReward speichern
        self.totalReward += reward
        
        #Observation und Info speichern
        obs = self.get_obs()
        info = {"area": float(currArea), "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated), "step_ctr": self.step_ctr}
        
        return obs, reward, terminated, truncated, info
        
        
        
        