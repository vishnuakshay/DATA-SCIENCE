from rasa_nlu.model import Interpreter
import json
from pathlib import Path

model_directory = './models/default/model_20180914-101429'
nlu_interpreter = Interpreter.load(model_directory)

capitals_json = Path("capitals.json").read_text()
capitals = json.loads(capitals_json)


def parse_message(text):
    data = nlu_interpreter.parse(text)
    intent_name = data["intent"]["name"]
    confidence = data["intent"]["confidence"]
    entities = data["entities"]

    print(f"- Model detected intent: {intent_name} ({confidence})")

    if confidence > 0.3:
        return intent_name, entities
    else:
        return "not_understood", []


def handle_message(text):
    intent_name, entities = parse_message(text)
    response = answer_question(intent_name, entities)

    # Respond to the user
    print(response)


def answer_question(intent_name, entities):
    response = "Huh?"

    if intent_name == "hello":
        response = "Hello! I'm a Bot."
    elif intent_name == "lunch_idea":
        response = "Let's eat something healthy for once!"
    elif intent_name == "office_hours":
        response = "The office opens at 8am and closes at 6pm."
    elif intent_name == "office_location":
        response = "The office is at 123 Machine Learning Street."
    elif intent_name == "wifi_password":
        response = "The office wifi password is 'abc123."
    elif intent_name == "capital_lookup":
        response = get_capital(entities)

    return response


def get_capital(entities):
    if len(entities) == 0:
        response = "Where are you asking about exactly?"
    else:
        region = entities[0]["value"].title()
        if region in capitals:
            response = f"The capital of {region} is {capitals[region]}"
        else:
            response = f"I'm not sure what the capital of {region} is."

    return response


while True:
    user_text = input(" > ")
    handle_message(user_text)