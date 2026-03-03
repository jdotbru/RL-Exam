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
        self.useMaxSteps = False
        self.useAntagonist = bool(givenUseAntagonist)
        self.maxSteps = 500
        self.minSteps = 50
    
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
        
        #Action-space
        if self.useAntagonist:
            self.action_space = spaces.Dict({
                "protagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "antagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            })
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        #Observation-space
        obs_low = np.array([-self.degree_limit_rad]*3 + [-np.inf]*2, dtype=np.float32)
        obs_high = np.array([self.degree_limit_rad]*3 + [np.inf]*2, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    
    #---------------------    
    #---Hilfsfunktionen---
    #---------------------
    def get_obs(self) -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        return np.concatenate([self.joint_angles, self.currPos]).astype(np.float32)
     
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
    
    def getMaxArea(self) -> float:
        #Identifiziert die drei Eckpunkte des größtmöglichen Dreiecks aus den besuchten punkten des Fingers. Diese Eckpunkte werden in "corners" gespeichert.
        
        #TO-DO: Duplikate entfernen...
        
        #Prüfen, ob genug Punkte gesammelt wurden
        points = self.positionSaver
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
        #gibt True zurück wenn der aktuelle Zustand auf maximal einen Millimetter an den Startzustand herankommt und die Mindestanzahl Schritte ausgeführt wurde
        distance = math.sqrt((self.currPos[0] - self.startPos[0])**2 + (self.currPos[1] - self.startPos[1])**2)
        if (distance <= 1 and self.step_ctr > self.minSteps):
            return True
        return False
    
    def isTruncated(self) ->bool:
        #gibt True zurück wenn die maximale Anzahl an Schritten überschritten wurde
        return self.step_ctr >= self.maxSteps
      
    #-------------------
    #---Gymnasium API---
    #-------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        #Counter und Speicher zurücksetzen
        self.step_ctr = 0
        self.positionSaver = []
        
        #Initiale Gelenkwinkel setzen (zufällig klein um 0)
        self.joint_angles = self.rng.uniform(low=-0.1, high=0.1, size=(3,))
        
        #Positionen und History initialisieren
        self.currPos = self.calculate_new_Position(self.joint_angles)
        
        self.startPos = self.currPos.copy()
        
        #Observation & Info zurückgeben
        obs = self.get_obs()
        info = {"area": 0.0, "closed": False,}
        
        return obs, info
    
    def step(self, action):
        '''
        Vorgehen pro Step: 
        - Action: Abweichung je Gelenk zwischen -1 und 1 Grad -> übergeben von Agent
        - Beenden der Sequenz wenn Mindestanzahl Schritte gelaufen wurde und Finger sich wieder in Range von 1 Distanzeinheit um den Startpunkt befindet
        - Außerdem Beenden der Sequenz wenn Maximalanzahl Schritte gelaufen wurde
        - 
        '''
        #Step-Counter erhöhen
        self.step_ctr += 1
        
        if self.useAntagonist:
            actionPro = np.asarray(action["protagonist"])
        else:
            actionPro = np.asarray(action)
            
        #Action in Delta der Einheit Rad interpretieren
        delta = actionPro * (math.pi / 180.0)
        self.joint_angles += delta
        
        #Prüfen, ob ein Gelenk überdreht
        if np.any(self.joint_angles > self.degree_limit_rad) or np.any(self.joint_angles < -self.degree_limit_rad):
            print("Diese Aktion kann nicht ausgeführt werden, da eines der Gelenke überdrehen würde: ")
            for x in self.joint_angles:
                print(x, ", ")
            self.joint_angles -= delta
        
        #neue Position berechnen
        self.currPos = self.calculate_new_Position(self.joint_angles)
        self.positionSaver.append(self.currPos.copy())
        
        
        #Prüfen, ob Sequenz zuende ist
        terminated = False
        truncated = False
        reward = 0.0
        if self.isFinished():
            print("Die Sequenz wurde erfolgreich abgeschlossen.")
            reward = self.getMaxArea()
            terminated = True
        elif self.isTruncated():
            print("Die Sequenz wurde abgeschlossen, war allerdings nicht erfolgreich.")
            truncated = True
        
        obs = self.get_obs()
        info = {"area": float(reward), "terminated": bool(terminated), "step_ctr": self.step_ctr}
        
        return obs, reward, terminated, truncated, info
        
        
        
        