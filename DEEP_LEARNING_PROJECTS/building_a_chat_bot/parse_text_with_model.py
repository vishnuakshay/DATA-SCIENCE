import json
from rasa_nlu.model import Interpreter

model_directory = './models/default/model_20180914-101429'
nlu_interpreter = Interpreter.load(model_directory)

text = "What is the capital of Great Britain?"
data = nlu_interpreter.parse(text)
print(json.dumps(data, indent=2))