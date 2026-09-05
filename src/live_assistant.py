# src/live_assistant.py
#
# Draait op de achtergrond terwijl je Project Zomboid speelt: neemt periodiek
# een screenshot, laat het model erop los, berekent de risicoscore en toont
# een waarschuwing. Gebruikt standaard de lokale, regelgebaseerde advisor
# (onbeperkt bruikbaar) -- GenAI kan optioneel af en toe gebruikt worden.
#
# Slaat ook elk geanalyseerd frame (met bounding boxes erop getekend) op in
# 'live_captures/', zodat je achteraf kan nakijken wat het model precies zag
# op het moment van elke waarschuwing.
#
# Stoppen: Ctrl+C in de terminal.

import time
import importlib
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

mss = importlib.import_module("mss")

from risk_score import compute_risk_score
from rule_based_advisor import generate_advice_local

MODEL_PATH = "models/zomboid_v1.pt"
CAPTURE_INTERVAL_SECONDS = 3        # hoe vaak een nieuwe screenshot genomen wordt
CONF_THRESHOLD = 0.2

# Zet op True om af en toe (bv. enkel bij hoog risico) ook GenAI-advies te vragen.
# Gezien de daglimiet van 5 aanvragen: hou dit spaarzaam.
USE_GENAI_ON_HIGH_RISK = False
GENAI_RISK_THRESHOLD = 40

# Elk geanalyseerd frame opslaan met bounding boxes erop getekend, zodat je
# achteraf kan nakijken wat het model zag. Zet op False om dit uit te schakelen
# (bv. bij lange sessies, om niet duizenden afbeeldingen op te stapelen).
SAVE_ANNOTATED_FRAMES = True
CAPTURES_DIR = Path("live_captures")

# Enkel frames bewaren vanaf dit risiconiveau (0 = alles bewaren). Handig om
# schijfruimte te besparen tijdens lange sessies en enkel de interessante
# momenten (hoog risico) achteraf te kunnen bekijken.
SAVE_ONLY_ABOVE_RISK = 0


def capture_screen(sct, monitor) -> np.ndarray:
    """Neemt een screenshot van het opgegeven scherm en zet het om naar een numpy array (BGR, zoals OpenCV verwacht)."""
    shot = sct.grab(monitor)
    frame = np.array(shot)[:, :, :3]  # BGRA -> BGR, alpha-kanaal weglaten
    return frame


def run_live_assistant():
    print("Zomboid Survival Assistant -- live modus gestart.")
    print(f"Model: {MODEL_PATH}")
    print(f"Interval: {CAPTURE_INTERVAL_SECONDS}s. Druk Ctrl+C om te stoppen.")

    if SAVE_ANNOTATED_FRAMES:
        CAPTURES_DIR.mkdir(exist_ok=True)
        print(f"Geannoteerde frames worden opgeslagen in: {CAPTURES_DIR.resolve()}\n")
    else:
        print()

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
                timestamp_filename = time.strftime("%Y%m%d_%H%M%S")
                richtingen = ", ".join(risk_data["safest_directions"]) if risk_data["safest_directions"] else "-"
                print(f"[{timestamp}] risk={risk_data['risk_score']:>6.2f}  "
                      f"richting(en)={richtingen}  |  {advice}")

                # Frame met bounding boxes opslaan voor latere controle
                if SAVE_ANNOTATED_FRAMES and risk_data["risk_score"] >= SAVE_ONLY_ABOVE_RISK:
                    annotated = result.plot()  # numpy array (BGR) met boxes/labels erop getekend
                    filename = CAPTURES_DIR / f"{timestamp_filename}_risk{risk_data['risk_score']:.0f}.jpg"
                    cv2.imwrite(str(filename), annotated)

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
            if SAVE_ANNOTATED_FRAMES:
                saved_count = len(list(CAPTURES_DIR.glob("*.jpg")))
                print(f"{saved_count} geannoteerde frames opgeslagen in {CAPTURES_DIR.resolve()}")


if __name__ == "__main__":
    run_live_assistant()