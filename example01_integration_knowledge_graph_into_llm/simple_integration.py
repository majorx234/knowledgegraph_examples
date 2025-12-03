from transformers import pipeline, AutoTokenizer
import torch
import re

# device config check cuda cpu
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
    )

# some qa model
# https://huggingface.co/deepset/gelectra-base-germanquad
model_name = "deepset/gelectra-base-germanquad"

# pipeline object is generated
qa_pipeline = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=model_name,
    device=0 if device != "cpu" else -1
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

        # use q/a-pipeline: question + context (summary)
        result = qa_pipeline(question=question, context=summary)
        return result["answer"]
    else:
        return "Ich habe keine Informationen zu diesem Himmelsobjekt."


if __name__ == "__main__":
    # Beispielanfrage
    question = "Wie weit ist Sirius von der Erde entfernt?"
    answer = ask_astronomy_question(question)
    print("Antwort:", answer)
