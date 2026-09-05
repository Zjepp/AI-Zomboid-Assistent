from ultralytics import YOLO
from pathlib import Path
import math

CLASS_NAMES = ['Door', 'Fire', 'Player', 'Tree', 'Zombie', 'Zombie Dead', 'window']

MAX_DISTANCE = math.sqrt(2) 

ISO_OFFSET_DEGREES = 45

COMPASS_DIRECTIONS = ["N", "NO", "O", "ZO", "Z", "ZW", "W", "NW"]


def proximity_from_distance(distance: float) -> float:
    """
    Zet een genormaliseerde afstand (0 = zelfde plek, ~1.41 = tegenovergestelde hoeken)
    om naar een 'dichtbij-gewicht' tussen 0 en 1. Dichterbij = hoger gewicht.
    """
    weight = 1.0 - (distance / MAX_DISTANCE)
    return max(0.0, weight)


def proximity_from_box_proxy(box_area: float, y_center: float) -> float:
    """
    Fallback-proxy voor 'dichtbij' wanneer er geen speler-box gedetecteerd is:
    grotere box + lager in beeld = dichterbij.
    """
    return box_area * (0.5 + 0.5 * y_center)


def screen_bearing_to_compass(dx: float, dy: float) -> tuple[str, float]:
    """
    Zet een schermrichting (dx, dy vanaf de speler) om naar een spelkompas-richting,
    rekening houdend met de isometrische camerarotatie.

    Schermcoördinaten: dx > 0 = rechts, dy > 0 = naar beneden (normale beeldconventie).
    """
    screen_angle = math.degrees(math.atan2(-dy, dx))
    screen_angle = (screen_angle + 360) % 360

    angle_from_top = (90 - screen_angle) % 360

    compass_angle = (angle_from_top + ISO_OFFSET_DEGREES) % 360

    index = round(compass_angle / 45) % 8
    return COMPASS_DIRECTIONS[index], compass_angle


def compute_safest_direction(detections: list[dict], player_pos: tuple[float, float] | None) -> dict:
    """
    Bepaalt, op basis van de posities van zombies/vuur relatief aan de speler,
    welke van de 8 kompasrichtingen het minst bedreigd zijn (= veiligste richtingen om heen te gaan).

    Geeft alle richtingen terug die (binnen een kleine tolerantie) gelijk scoren aan het
    minimum, in plaats van willekeurig de eerste in de lijst te kiezen bij gelijkstand.
    """
    if player_pos is None:
        return {"safest_directions": [], "safest_direction": None, "direction_scores": {}}

    threat_by_direction = {d: 0.0 for d in COMPASS_DIRECTIONS}

    for det in detections:
        if det["class"] not in ("Zombie", "Fire"):
            continue

        dx = det["x"] - player_pos[0]
        dy = det["y"] - player_pos[1]
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            continue

        direction, _ = screen_bearing_to_compass(dx, dy)
        weight = proximity_from_distance(distance)

        multiplier = 1.5 if det["class"] == "Fire" else 1.0
        threat_by_direction[direction] += weight * multiplier

    min_threat = min(threat_by_direction.values())
    tolerance = 1e-6
    safest_directions = [d for d, v in threat_by_direction.items() if v <= min_threat + tolerance]

    return {
        "safest_directions": safest_directions,
        "safest_direction": safest_directions[0] if safest_directions else None,
        "direction_scores": {k: round(v, 3) for k, v in threat_by_direction.items()},
    }


def compute_risk_score(result) -> dict:
    """
    Berekent een risicoscore op basis van gedetecteerde objecten in één screenshot.

    Als de speler zelf gedetecteerd is, wordt de echte relatieve afstand
    (speler-positie tot object-positie) gebruikt in plaats van de ruwe proxy,
    en wordt ook de veiligste kompasrichting(en) berekend.
    """
    detections = []
    player_pos = None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id]
        x_center, y_center, w, h = box.xywhn[0].tolist()
        box_area = w * h

        detections.append({
            "class": cls_name,
            "x": x_center,
            "y": y_center,
            "area": box_area,
        })

        if cls_name == "Player" and player_pos is None:
            player_pos = (x_center, y_center)

    zombie_score = 0.0
    fire_score = 0.0
    vegetation_coverage = 0.0
    nearest_exit_score = 0.0
    zombie_dead_count = 0

    zones = {"left": False, "center": False, "right": False}

    for det in detections:
        cls_name = det["class"]
        x_center, y_center, box_area = det["x"], det["y"], det["area"]

        if player_pos is not None:
            distance = math.dist((x_center, y_center), player_pos)
            proximity_weight = proximity_from_distance(distance)
        else:
            proximity_weight = proximity_from_box_proxy(box_area, y_center)

        if cls_name == "Zombie":
            zombie_score += proximity_weight * 10

            if x_center < 0.33:
                zones["left"] = True
            elif x_center < 0.66:
                zones["center"] = True
            else:
                zones["right"] = True

        elif cls_name == "Zombie Dead":
            zombie_dead_count += 1

        elif cls_name == "Fire":
            fire_score += proximity_weight * 15

        elif cls_name == "Tree":
            vegetation_coverage += box_area

        elif cls_name in ("Door", "window"):
            nearest_exit_score = max(nearest_exit_score, proximity_weight)

    blocked_zones = sum(zones.values())
    free_routes = 3 - blocked_zones

    visibility_penalty = vegetation_coverage * 5
    exit_bonus = nearest_exit_score * 3

    total_risk = (
        zombie_score
        + fire_score
        + visibility_penalty
        + zombie_dead_count * 0.1
        - exit_bonus
        - (free_routes * 2)
    )
    total_risk = max(0.0, total_risk)

    direction_info = compute_safest_direction(detections, player_pos)

    return {
        "risk_score": round(total_risk, 2),
        "zombie_score": round(zombie_score, 2),
        "fire_score": round(fire_score, 2),
        "zombie_dead_count": zombie_dead_count,
        "zombie_dead_score": round(zombie_dead_count * 0.2, 2),
        "visibility_penalty": round(visibility_penalty, 2),
        "exit_bonus": round(exit_bonus, 2),
        "free_routes": free_routes,
        "player_detected": player_pos is not None,
        "safest_directions": direction_info["safest_directions"],
        "safest_direction": direction_info["safest_direction"],
        "direction_scores": direction_info["direction_scores"],
    }


def analyze_screenshot(image_path: str, model_path: str = "models/zomboid_v1.pt"):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    result = results[0]
    return compute_risk_score(result)


def analyze_folder(folder_path: str, model_path: str = "models/zomboid_v1.pt") -> list[dict]:
    model = YOLO(model_path)
    results = model.predict(source=folder_path, conf=0.25, verbose=False)

    all_risk_data = []
    for result in results:
        risk_data = compute_risk_score(result)
        risk_data["image"] = Path(result.path).name
        all_risk_data.append(risk_data)

    return all_risk_data


if __name__ == "__main__":
    import sys

    path_arg = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/test/images"
    target_path = Path(path_arg)

    if target_path.is_dir():
        all_results = analyze_folder(str(target_path))
        for r in all_results:
            richtingen = ", ".join(r["safest_directions"]) if r["safest_directions"] else "-"
            print(f"{r['image']}: risk_score={r['risk_score']}  richting(en)={richtingen}")

        scores = [r["risk_score"] for r in all_results]
        if scores:
            print(f"\n{len(scores)} afbeeldingen geanalyseerd.")
            print(f"Gemiddelde risk_score: {sum(scores) / len(scores):.2f}")
            print(f"Min: {min(scores):.2f}  Max: {max(scores):.2f}")
    else:
        risk = analyze_screenshot(str(target_path))
        print(risk)