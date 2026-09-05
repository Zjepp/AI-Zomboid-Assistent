# src/rule_based_advisor.py
#
# Genereert waarschuwingen zonder een externe API aan te roepen -- volledig
# lokaal, dus geen limiet op het aantal keer dat dit per sessie draait.
# Bedoeld als het standaard-pad voor de live-assistent; GenAI (genai_advisor.py)
# kan optioneel apart ingezet worden voor een uitgebreidere, minder frequente samenvatting.

DIRECTION_NAMES_NL = {
    "N": "het noorden",
    "NO": "het noordoosten",
    "O": "het oosten",
    "ZO": "het zuidoosten",
    "Z": "het zuiden",
    "ZW": "het zuidwesten",
    "W": "het westen",
    "NW": "het noordwesten",
}


def generate_advice_local(risk_data: dict) -> str:
    """
    Bouwt een korte waarschuwing op basis van vaste regels/drempelwaarden,
    zonder een taalmodel aan te spreken.
    """
    risk = risk_data["risk_score"]
    zombie_score = risk_data["zombie_score"]
    fire_score = risk_data["fire_score"]
    zombie_dead_count = risk_data["zombie_dead_count"]
    free_routes = risk_data["free_routes"]
    safest_direction = risk_data.get("safest_direction")
    player_detected = risk_data["player_detected"]

    parts = []

    # Risiconiveau als kop
    if risk >= 40:
        parts.append("GEVAAR HOOG.")
    elif risk >= 15:
        parts.append("Waarschuwing:")
    elif risk > 0:
        parts.append("Lichte dreiging:")
    else:
        parts.append("Veilig.")

    # Zombie-dreiging
    if zombie_score >= 30:
        parts.append("Een grote groep zombies is dichtbij.")
    elif zombie_score >= 10:
        parts.append("Er zijn zombies in de buurt.")
    elif zombie_score > 0:
        parts.append("Enkele zombies op afstand.")

    # Vuur, apart genoemd want zwaarder risico
    if fire_score >= 10:
        parts.append("Vuur vormt een acuut gevaar dichtbij!")
    elif fire_score > 0:
        parts.append("Er is vuur zichtbaar in de omgeving.")

    # Lijken (indirect risico)
    if zombie_dead_count >= 15:
        parts.append("Veel kadavers dichtbij -- blijf hier niet te lang, kans op ziekte.")

    # Richting en ontsnappingsroutes
    if player_detected and safest_direction is not None:
        richting_nl = DIRECTION_NAMES_NL.get(safest_direction, safest_direction)
        if risk >= 15:
            parts.append(f"Veiligste richting om te vluchten: {richting_nl}.")
        elif free_routes > 0:
            parts.append(f"Indien nodig, {richting_nl} lijkt het minst bedreigd.")
    elif not player_detected:
        parts.append("(Speler niet gedetecteerd in beeld -- richtingsadvies onbetrouwbaar.)")

    if free_routes == 0 and risk >= 15:
        parts.append("Let op: weinig vrije ontsnappingsroutes zichtbaar.")

    return " ".join(parts)


if __name__ == "__main__":
    # Snelle test met voorbeelddata
    test_risk = {
        "risk_score": 41.5,
        "zombie_score": 44.0,
        "fire_score": 0.0,
        "zombie_dead_count": 34,
        "zombie_dead_score": 6.8,
        "visibility_penalty": 0.07,
        "exit_bonus": 1.97,
        "free_routes": 2,
        "player_detected": True,
        "safest_direction": "NW",
        "direction_scores": {},
    }
    print(generate_advice_local(test_risk))
