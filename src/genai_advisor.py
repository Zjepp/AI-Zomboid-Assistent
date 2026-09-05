import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_advice(risk_data: dict) -> str:
    prompt = f"""Je bent een survival-assistent voor het spel Project Zomboid.
Op basis van deze analyse van de huidige situatie van de speler, geef een korte,
directe waarschuwing of advies in het Nederlands. Wees to the point, zoals een spanningsvol
in-game bericht.
Leg ook uit waarom en op wat je je gebaseerd hebt.

Dode zombies zijn geen directe bedreiging, maar kunnen ziekte veroorzaken als de speler er
te lang te dicht bij blijft.

BELANGRIJKE BEPERKING: het systeem weet niet of de speler zich momenteel binnen of buiten
bevindt, en houdt geen rekening met muren of andere blokkades. Enkel de aanwezigheid van
ramen en deuren in beeld wordt geteld als mogelijke ontsnappingsroute. Vermijd daarom
concrete aannames zoals "klim door het raam" of "ren naar buiten", formuleer in plaats
daarvan generiek, bijvoorbeeld "gebruik een nabije uitgang" of "houd je vluchtroute in de
gaten", en vermeld expliciet dat de speler zelf moet inschatten of de omgeving klopt met dit advies.
 

Data:
- Risicoscore: {risk_data['risk_score']}
- Zombie-dreiging: {risk_data['zombie_score']}
- Dode zombies-risico: {risk_data['zombie_dead_score']}
- Zichtbaarheid-penalty (vegetatie): {risk_data['visibility_penalty']}
- Aantal vrije ontsnappingsroutes: {risk_data['free_routes']}

Geef enkel het advies, geen inleiding."""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text.strip()


if __name__ == "__main__":
    from risk_score import analyze_screenshot

    risk = analyze_screenshot("data/dataset/test/images/img_133_jpg.rf.5EOwFgcvBGe6UX9hETEJ.jpg")
    advice = generate_advice(risk)
    print(f"Risico data: {risk}")
    print(f"Advies: {advice}")