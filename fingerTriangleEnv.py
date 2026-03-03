import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any
import math
from itertools import combinations

class FingerTriangleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    #einstellbare Parameter
    useMaxSteps = False
    useAntagonist = False
    
    link_lengths = np.array([5.0, 2.5, 2.5])
    maxSteps = 500
    step_ctr = 0
    rng = None
    degree_limit_rad = pi/2
    seed = 0
    joint_angles = np.zeros(3)
    positionSaver = []
    currPos = np.zeros(2)
    start_pos = np.zeros(2)
    corners = []
    
    def __init__(self, givenUseAntagonist):
        super().__init__()
        useAntagonist = givenUseAntagonist
        
        #Random Number Generator deterministisch/reproduzierbar initialisieren
        rng = np.random.default_rng(seed)
        
        #Action-space
        action_space = spaces.Dict({
            "protagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
            "antagonist": spaces.Box(low=-1.0, high=1.0, shape=(3,))
        })
        
        #Observation-space
        obs_low = np.array([-degree_limit_rad]*3 + [-np.inf]*6)
        obs_high = np.array([degree_limit_rad]*3 + [np.inf]*6)
        observation_space = spaces.Box(low=obs_low, high=obs_high)
        
        #Interner Zustand
        stepCounter = 0
        joint_angles = np.zeros(3)
        currPos = np.zeros(2)
        
        #Ecken für das Dreieck
        corners = []
    
    #---------------------    
    #---Hilfsfunktionen---
    #---------------------
    def get_obs() -> np.ndarray:
        #Beobachtungsvektor zusammenbauen
        return np.concatenate([joint_angles, currPos])
     
    def calculate_new_Position(newAngles: np.ndarray) -> np.ndarray:
        #berechnet neue Position des Fingers
        p1, p2, p3 = link_lengths
        a1, a2, a3 = newAngles
        
        b1 = a1
        b2 = a1 + a2
        b3 = a1 + a2 + a3
        
        x = p1 * np.cos(b1) + p2 * np.cos(b2) + p3 * np.cos(b3)
        y = p1 * np.sin(b1) + p2 * np.sin(b2) + p3 * np.sin(b3)
        return np.array([x, y])
    
    def getMaxArea() -> int:
        #Identifiziert die drei Eckpunkte des größtmöglichen Dreiecks aus den besuchten punkten des Fingers. Diese Eckpunkte werden in "corners" gespeichert.
        
        #TO-DO: Duplikate entfernen...
        
        #Prüfen, ob genug Punkte gesammelt wurden
        if positionSaver.size() < 3:
            corners = []
            return np.zeros((0, 2))
        
        #Alle möglichen Kombinationen generieren und Ecken des größtmöglichen Dreiecks finden
        maxArea = 0
        
        for p1, p2, p3 in combinations(points, 3):
            area = calculateTriangleArea(p1, p2, p3)
            if area > maxArea:
                maxArea = area
        return maxArea
    
    def calculateTriangleArea(corner1: np.ndarray, corner2: np.ndarray, corner3: np.ndarray) -> float:
        #Flächenformel: 0,5 * |det(corner2-corner1, corner3-corner1)|
        
        v1 = corner2 - corner1
        v2 = corner3 - corner1
        det = v1[0]*v2[1] - v1[1]*v2[0]
        return float(0.5 * abs(det))
    
    def isFinishedSuccesful() -> bool:
        #gibt True zurück wenn der aktuelle Zustand auf maximal einen Millimetter an den Startzustand herankommt
        distance = math.sqrt((currPos[0] - startPos[0])**2 + (currPos[1] - startPos[1])**2)
        if (distance <= 1):
            return True
        return False
    
    def isFinishedUnsuccesfull() ->bool:
        #gibt True zurück wenn die maximale Anzahl an Schritten überschritten wurde
        return step_ctr >= maxSteps
      
    #-------------------
    #---Gymnasium API---
    #-------------------
    def reset() -> Tuple[np.ndarray, Dict[str, Any]]:
        #step-counter zurücksetzen
        step_ctr = 0
        
        #Initiale Gelenkwinkel setzen (zufällig klein um 0)
        joint_angles = rng.uniform(low=-0.1, high=0.1, size=(3,))
        
        #Positionen und History initialisieren
        pos = calculate_new_Position(joint_angles)
        
        corners = []
        start_pos = currPos.copy()
        
        #Observation & Info zurückgeben
        obs = get_obs()
        info = {"corners": [], "area": 0.0, "closed": False,}
        
        return obs, info
    
    def step(actionPro: np.ndarray):
        '''
        Vorgehen pro Step: 
        - Action: Abweichung je Gelenk zwischen -1 und 1 Grad -> übergeben von Agent
        - Beenden der Sequenz wenn Mindestanzahl Schritte gelaufen wurde und Finger sich wieder in Range von 1 mm um den Startpunkt befindet
        - Außerdem Beenden der Sequenz wenn Maximalanzahl Schritte gelaufen wurde
        - 
        '''
        #Step-Counter erhöhen
        step_ctr += 1
        
        #neue Position berechnen
        newPos = calculate_new_Position(actionPro)
        positionSaver.append(newPos)
        
        
        #Prüfen, ob Sequenz zuende ist
        if isFinishedSuccesfull():
            println("Die Sequenz wurde erfolgreich abgeschlossen.")
            reward = getMaxArea()
            terminated = True
        elif isFinishedUnsuccesfull():
            println("Die Sequenz wurde abgeschlossen, war allerdings nicht erfolgreich.")
            reward = 0.0
            terminated = True
        else:
            reward = 0.0
            terminated = False
        
        return reward, terminated
        
        
        
        