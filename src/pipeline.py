from risk_score import analyze_screenshot
from genai_advisor import generate_advice
import sys

def run_pipeline(image_path: str):
    risk_data = analyze_screenshot(image_path)
    # advice = generate_advice(risk_data)

    print(f"\n--- Analyse voor {image_path} ---")
    print(f"Risicoscore: {risk_data['risk_score']}")
    print(f"  Zombie-dreiging: {risk_data['zombie_score']}")
    print(f"  Zichtbaarheid-penalty: {risk_data['visibility_penalty']}")
    print(f"  Vrije ontsnappingsroutes: {risk_data['free_routes']}")
    # print(f"\nAdvies: {advice}\n")

    return risk_data, #advice

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset/test/images/img_001.jpg"
    run_pipeline(image_path)
