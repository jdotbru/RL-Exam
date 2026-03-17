#Unterstützung von KI in Design-Entscheidungen und teilweise bei Codeteilen

from typing import Optional

import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    #berechnet Winkel zwischen zwei Geraden
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm == 0:
        return 0.0

    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_new_position(angles: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
    #berechnet aus den Gelenkwinkeln die Fingerspitzenposition
    p1, p2, p3 = link_lengths
    a1, a2, a3 = np.deg2rad(angles)

    b1 = a1
    b2 = a1 + a2
    b3 = a1 + a2 + a3

    x = p1 * np.cos(b1) + p2 * np.cos(b2) + p3 * np.cos(b3)
    y = p1 * np.sin(b1) + p2 * np.sin(b2) + p3 * np.sin(b3)
    return np.array([x, y], dtype=np.float32)


def calculate_triangle_area(corner1: np.ndarray, corner2: np.ndarray, corner3: np.ndarray) -> float:
    #berechnet die Fläche eines Dreiecks
    #Formel: 0,5 * [det(corner2 - corner1, corner3 – corner 1)]
    v1 = corner2 - corner1
    v2 = corner3 - corner1
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return float(0.5 * abs(det))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    #Berechnet Vektorrichtung auf Werte unter 1, sodass nur Richtung statt Länge entscheidet
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    #Berechnet die Differenz eines Punktes zu einer Linie
    segment = end - start
    seg_norm = np.linalg.norm(segment)
    if seg_norm < 1e-6:
        return float(np.linalg.norm(point - start))

    rel = point - start
    projection = np.dot(rel, segment) / (seg_norm ** 2)
    projection = np.clip(projection, 0.0, 1.0)
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def segment_mean_deviation(start: np.ndarray, end: np.ndarray, points: list[np.ndarray]) -> float:
    #Berechnet durchschnittliche Abweichung aller Punkte auf einer Kante zu der Linie zwischen Start und Endpunkt
    if len(points) <= 2:
        return 0.0
    distances = [point_line_distance(point, start, end) for point in points[1:-1]]
    if not distances:
        return 0.0
    return float(np.mean(distances))


def is_corner(env) -> bool:
    #Bildet Kanten ausgehend von einem Punkt zu Punkten vor und nach dem Punkt, berechnet den Winkel und entscheidet, ob es eine Ecke ist
    if len(env.positionSaver) < 2 * env.cornerWindow + 1:
        return False

    p1 = env.positionSaver[-(2 * env.cornerWindow + 1)]
    p2 = env.positionSaver[-(env.cornerWindow + 1)]
    p3 = env.positionSaver[-1]

    v1 = p2 - p1
    v2 = p3 - p2

    if np.linalg.norm(v1) < env.min_segment_len or np.linalg.norm(v2) < env.min_segment_len:
        return False

    angle = angle_between(v1, v2)
    return env.angleThresholdMax > angle > env.angleThresholdMin


def get_triangle_edge_lengths(env) -> tuple[float, float, float]:
    #berechnet Länge aller Kanten
    if env.corner1 is None or env.corner2 is None:
        return 0.0, 0.0, 0.0
    edge1 = float(np.linalg.norm(env.corner1 - env.startPos))
    edge2 = float(np.linalg.norm(env.corner2 - env.corner1))
    edge3 = float(np.linalg.norm(env.startPos - env.corner2))
    return edge1, edge2, edge3


def get_triangle_edge_metrics(env) -> tuple[float, float]:
    #Gibt durchschnittliche Kantenlänge und Kantengewichtung zurück
    edge1, edge2, edge3 = get_triangle_edge_lengths(env)
    edges = np.array([edge1, edge2, edge3], dtype=np.float32)
    mean_edge_length = float(np.mean(edges))
    max_edge_length = float(np.max(edges))
    if max_edge_length <= 1e-6:
        return mean_edge_length, 0.0
    edge_balance = float(np.min(edges) / max_edge_length)
    return mean_edge_length, edge_balance


def get_segment_points(env, start_idx: int, end_idx: int) -> list[np.ndarray]:
    #Gibt alle Punkte einer Phase zurücl
    return [np.asarray(point) for point in env.positionSaver[start_idx : end_idx + 1]]


def evaluate_triangle_shape(env) -> tuple[float, float, float]:
    #Gibt Fläche, durchschnittliche Abweichung von der Kante und den Straightness Score zurück
    if env.corner1 is None or env.corner2 is None or env.corner1_idx is None or env.corner2_idx is None:
        return 0.0, 0.0, 0.0

    seg1 = get_segment_points(env, 0, env.corner1_idx)
    seg2 = get_segment_points(env, env.corner1_idx, env.corner2_idx)
    seg3 = get_segment_points(env, env.corner2_idx, len(env.positionSaver) - 1)

    if len(seg1) < 2 or len(seg2) < 2 or len(seg3) < 2:
        return 0.0, 0.0, 0.0

    dev1 = segment_mean_deviation(env.startPos, env.corner1, seg1)
    dev2 = segment_mean_deviation(env.corner1, env.corner2, seg2)
    dev3 = segment_mean_deviation(env.corner2, env.startPos, seg3)
    mean_deviation = float(np.mean([dev1, dev2, dev3]))

    area = calculate_triangle_area(env.startPos, env.corner1, env.corner2)
    straightness_score = max(0.0, 1.0 - mean_deviation / env.maxMeanLineDeviation)
    return float(area), mean_deviation, float(straightness_score)


def is_good_triangle(env) -> bool:
    #Vergleicht Metriken des Dreiecks mit Schwellwerten und definiert, ob das Dreieck als gut angesehen werden kann
    if env.corner1 is None or env.corner2 is None or env.currPhase != 2:
        return False

    area, _, straightness_score = evaluate_triangle_shape(env)
    mean_edge_length, edge_balance = get_triangle_edge_metrics(env)
    return bool(
        area >= env.minArea
        and straightness_score >= env.minStraightnessForSuccess
        and env.extraCornerCount <= env.maxExtraCornersForSuccess
        and mean_edge_length >= env.minMeanEdgeLengthForGoodTriangle
        and edge_balance >= env.minEdgeBalanceForGoodTriangle
    )


def calculate_turn_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    #Berechnet Winkel an einem Punkt
    v1 = p1 - p2
    v2 = p3 - p2
    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
        return angle_between(v1, v2)
    return 0.0


def get_current_move_direction(env) -> np.ndarray:
    #Gibt Richtung anhand der letzten zwei Punkte zurück
    if len(env.positionSaver) < 2:
        return np.zeros(2, dtype=np.float32)
    return normalize_vector(env.positionSaver[-1] - env.positionSaver[-2])


def maybe_initialize_phase_target_direction(env) -> None:
    #Gibt Richtung zurück zum Start vor wenn der Algorithmus sich in Phase 2 befindet
    if env.currPhase == 2 and env.corner2 is not None and env.phase2_target_direction is None:
        direction = env.startPos - env.corner2
        if np.linalg.norm(direction) >= env.min_segment_len:
            env.phase2_target_direction = normalize_vector(direction)


def get_corner2(env) -> Optional[np.ndarray]:
    #Gibt Ecke 2 zurück
    if env.currPhase == 2:
        return env.corner2
    return None


def get_phase_target_direction(env) -> np.ndarray:
    #Gibt Zielrichtung in Phase 2 zurück
    if env.currPhase == 2:
        if env.phase2_target_direction is not None:
            return env.phase2_target_direction
        if env.corner2 is not None:
            return normalize_vector(env.startPos - env.corner2)
    return np.zeros(2, dtype=np.float32)


def get_phase_line_deviation(env, position: Optional[np.ndarray] = None) -> float:
    #Gibt Abweichung eines Punktes in Phase 2 zurück
    anchor = get_corner2(env)
    direction = get_phase_target_direction(env)
    if anchor is None or np.linalg.norm(direction) < 1e-6:
        return 0.0
    position = env.currPos if position is None else position
    line_end = anchor + 20.0 * direction
    return point_line_distance(position, anchor, line_end)


def get_phase_progress(env, position: Optional[np.ndarray] = None) -> float:
    #Gibt Fortschritt in Richtung des Ziels in Phase 2 zurück
    anchor = get_corner2(env)
    direction = get_phase_target_direction(env)
    if anchor is None or np.linalg.norm(direction) < 1e-6:
        return 0.0
    position = env.currPos if position is None else position
    return float(max(0.0, np.dot(position - anchor, direction)))


def get_effective_closure_radius(env) -> float:
    #Gibt effektiven Abschlussradius auf Basis der Kantenlängen zurück
    if env.corner1 is None or env.corner2 is None:
        return float(env.closureRadius)

    edge1 = np.linalg.norm(env.corner1 - env.startPos)
    edge2 = np.linalg.norm(env.corner2 - env.corner1)
    edge3 = np.linalg.norm(env.startPos - env.corner2)
    mean_edge_length = float(np.mean([edge1, edge2, edge3]))

    dynamic_radius = env.closureRadiusRatio * mean_edge_length
    return float(np.clip(dynamic_radius, env.closureRadiusMin, env.closureRadiusMax))


def get_return_line_deviation(env, position: Optional[np.ndarray] = None) -> float:
    #Gibt Abweichung in Phase 2 zurück
    if env.corner2 is None:
        return 0.0
    position = env.currPos if position is None else position
    return point_line_distance(position, env.corner2, env.startPos)


def get_return_direction_alignment(env, prev_pos: np.ndarray, curr_pos: np.ndarray) -> float:
    #Misst wie nah die aktuelle Richtung mit der Zielrichtung übereinstimmt
    if env.corner2 is None:
        return 0.0
    move_dir = normalize_vector(curr_pos - prev_pos)
    target_dir = normalize_vector(env.startPos - env.corner2)
    return float(np.dot(move_dir, target_dir))


def update_best_form_snapshot(env) -> None:
    #Prüft ob ein neues bestes Dreieck gebildet wurde und speichert es wenn ja
    if env.currPhase != 2 or env.corner1 is None or env.corner2 is None:
        return

    area, mean_deviation, straightness_score = evaluate_triangle_shape(env)
    distance_to_start = float(np.linalg.norm(env.currPos - env.startPos))
    effective_closure_radius = get_effective_closure_radius(env)
    score = (
        135.0 * straightness_score
        + 20.0 * area
        - 18.0 * env.extraCornerCount
        - 34.0 * distance_to_start
    )
    if distance_to_start <= 1.25 * effective_closure_radius:
        score += 18.0
    if score <= env.bestFormScore:
        return

    env.bestFormScore = float(score)
    env.bestFormSnapshot = {
        "positions": np.array(env.positionSaver, dtype=np.float32).copy(),
        "area": float(area),
        "mean_line_deviation": float(mean_deviation),
        "triangle_straightness": float(straightness_score),
        "distance_to_start": float(distance_to_start),
        "extra_corners": int(env.extraCornerCount),
        "corner1": None if env.corner1 is None else np.array(env.corner1, dtype=np.float32).copy(),
        "corner2": None if env.corner2 is None else np.array(env.corner2, dtype=np.float32).copy(),
        "curriculum_stage": int(env.curriculum_stage),
        "effective_closure_radius": float(effective_closure_radius),
        "success_like": bool(
            distance_to_start <= effective_closure_radius
            and area >= env.minArea
            and env.step_ctr > env.minSteps
        ),
    }


def get_counted_extra_corners(env) -> int:
    #Gibt Anzahl von extra Ecken zurücl
    if env.curriculum_stage == 2:
        return min(env.extraCornerCount, env.stage2_maxCountedExtraCorners)
    return int(env.extraCornerCount)


def cap_late_stage_penalty(env, penalty: float, stage2_cap: float, stage3_cap: float) -> float:
    #Limitiert die Bestrafung zu späten Zeitpunkten, um die gelernte Policy nicht zu stark zu überschreiben
    if env.curriculum_stage == 2:
        return float(min(penalty, stage2_cap))
    if env.curriculum_stage >= 3:
        return float(min(penalty, stage3_cap))
    return float(penalty)
