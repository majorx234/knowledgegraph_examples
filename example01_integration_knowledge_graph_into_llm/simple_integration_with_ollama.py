import json
from ollama import Client
import re

OLLAMA_JSON_JSON_FILE="ollama_config.json"
ollama_json_file = open(OLLAMA_JSON_JSON_FILE)
ollama_config = json.load(ollama_json_file)
ollama_vars = {
    "host": ollama_config["base_url"],
    "model": ollama_config["model"]
}
client = Client(
  host=ollama_vars["host"],
  headers={'x-some-header': 'some-value'}
)

#simple knowledge graph as list of objects
knowledge_graph = [
    {
        "name": "Sonne",
        "type": "star",
        "distance_from_earth_ly": 0.00001581,
        "size_km": 1392700,
        "mass_kg": 1.989e30,
        "coordinates": {"ra": "00h 00m 00s", "dec": "+00° 00' 00\""}
    },
    {
        "name": "Sirius",
        "type": "star",
        "distance_from_earth_ly": 8.6,
        "size_km": 1.711e6,
        "mass_kg": 4.018e30,
        "coordinates": {"ra": "06h 45m 08.9s", "dec": "-16° 42' 58\""}
    },
    {
        "name": "Andromeda-Galaxie",
        "type": "galaxy",
        "distance_from_earth_ly": 2537000,
        "size_km": 220000,
        "mass_kg": 1.5e42,  # korrigierte Masse
        "coordinates": {"ra": "00h 42m 44.3s", "dec": "+41° 16' 09\""}
    },
    {
        "name": "Orion-Nebel",
        "type": "nebula",
        "distance_from_earth_ly": 1344,
        "size_km": 24,
        "mass_kg": 2e31,
        "coordinates": {"ra": "05h 35m 17.3s", "dec": "-05° 23' 28\""}
    },
    {
        "name": "Jupiter",
        "type": "planet",
        "distance_from_earth_ly": 0.000082,
        "size_km": 139820,
        "mass_kg": 1.898e27,
        "coordinates": {"ra": "18h 50m 00s", "dec": "-23° 00' 00\""}
    }
]


def get_astronomy_info(object_name):
    """
    iterate through knowledge graph array
    search for objects with name: object_name
    """
    for obj in knowledge_graph:
        if obj["name"].lower() == object_name.lower():
            return obj
    return None


def extract_object_name(question):
    """
    checks if in question is one
    of the object names of the knowledge graph
    uses iteration or regular expression
    """
    for obj in knowledge_graph:
        if obj["name"].lower() in question.lower():
            return obj["name"]
    # Fallback: regulärer Ausdruck
    match = re.search(r"\b(Sonne|Sirius|Andromeda-Galaxie|Orion-Nebel|↲Jupiter)\b", question, re.IGNORECASE)
    return match.group(0) if match else None


def ask_astronomy_question(question):
    """
    enhance question with facts out of the knowledge graph
    """
    object_name = extract_object_name(question)
    if not object_name:
        return "Ich konnte kein bekanntes Himmelsobjekt in der Frage finden."
    info = get_astronomy_info(object_name)
    if info:
        summary = (
            f"{info['name']} ist ein {info['type']}. "
            f"Es ist {info['distance_from_earth_ly']} Lichtjahre von der Erde entfernt. "
            f"Seine Größe beträgt {info['size_km']} km und seine Masse beträgt {info['mass_kg']} kg. "
            f"Seine Koordinaten sind RA: {info['coordinates']['ra']}, DEC: {info['coordinates']['dec']}."
        )
        # question + summary
        messages = [
            {
                'role': 'user',
                'content': 'Beantworte die Frage am Ende in deutsch mit Hilfe der folgenden Informationen. ' + summary +
                " Frage: " + question,
            },
        ]
        result = client.chat(model=ollama_vars['model'], messages=messages)
        return result['message']['content']
    else:
        return "Ich habe keine Informationen zu diesem Himmelsobjekt."


if __name__ == "__main__":
    # Beispielanfrage
    question = "Wie weit ist Sirius von der Erde entfernt?"
    answer = ask_astronomy_question(question)
    print("Antwort:", answer)
