# src/live_assistant.py
#
# Draait op de achtergrond terwijl je Project Zomboid speelt: neemt periodiek
# een screenshot, laat het model erop los, berekent de risicoscore en toont
# een waarschuwing. Gebruikt standaard de lokale, regelgebaseerde advisor
# (onbeperkt bruikbaar) -- GenAI kan optioneel af en toe gebruikt worden.
#
# Stoppen: Ctrl+C in de terminal.

import time
import numpy as np
import mss
from ultralytics import YOLO

from risk_score import compute_risk_score
from rule_based_advisor import generate_advice_local

MODEL_PATH = "models/zomboid_v1.pt"
CAPTURE_INTERVAL_SECONDS = 3        # hoe vaak een nieuwe screenshot genomen wordt
CONF_THRESHOLD = 0.25

# Zet op True om af en toe (bv. enkel bij hoog risico) ook GenAI-advies te vragen.
# Gezien de daglimiet van 5 aanvragen: hou dit spaarzaam.
USE_GENAI_ON_HIGH_RISK = False
GENAI_RISK_THRESHOLD = 40


def capture_screen(sct, monitor) -> np.ndarray:
    """Neemt een screenshot van het opgegeven scherm en zet het om naar een numpy array (BGR, zoals OpenCV verwacht)."""
    shot = sct.grab(monitor)
    frame = np.array(shot)[:, :, :3]  # BGRA -> BGR, alpha-kanaal weglaten
    return frame


def run_live_assistant():
    print("Zomboid Survival Assistant -- live modus gestart.")
    print(f"Model: {MODEL_PATH}")
    print(f"Interval: {CAPTURE_INTERVAL_SECONDS}s. Druk Ctrl+C om te stoppen.\n")

    model = YOLO(MODEL_PATH)

    with mss.MSS() as sct:
        # Standaard: hoofdscherm. Pas 'monitor' aan indien je meerdere schermen hebt
        # en het spel niet op het eerste scherm draait (sct.monitors[1] = eerste
        # fysieke scherm, sct.monitors[2] = tweede, etc. -- [0] is alle schermen samen).
        monitor = sct.monitors[1]

        last_genai_call_time = 0
        genai_cooldown_seconds = 300  # minimaal 5 minuten tussen GenAI-aanvragen

        try:
            while True:
                frame = capture_screen(sct, monitor)

                results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
                result = results[0]

                risk_data = compute_risk_score(result)
                advice = generate_advice_local(risk_data)

                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] risk={risk_data['risk_score']:>6.2f}  "
                      f"richting={risk_data['safest_direction']}  |  {advice}")

                # Optioneel: spaarzame GenAI-aanvraag bij hoog risico
                if (USE_GENAI_ON_HIGH_RISK
                        and risk_data["risk_score"] >= GENAI_RISK_THRESHOLD
                        and time.time() - last_genai_call_time > genai_cooldown_seconds):
                    try:
                        from genai_advisor import generate_advice
                        genai_advice = generate_advice(risk_data)
                        print(f"  [GenAI] {genai_advice}")
                        last_genai_call_time = time.time()
                    except Exception as e:
                        print(f"  [GenAI-oproep mislukt, ga verder met lokaal advies: {e}]")

                time.sleep(CAPTURE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nGestopt door gebruiker.")


if __name__ == "__main__":
    run_live_assistant()