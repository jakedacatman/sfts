from flask import Flask, jsonify, request
from datetime import datetime
from tensorflow.keras.models import load_model 
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)
prefix = "models/"

with open(f"{prefix}decision_tree.pkl", "rb") as file:
    model_tree = pickle.load(file)

model_nn = load_model(f"{prefix}classification_mlp.keras")

classes = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

#mappings for one-hot encoding
colors = {
    "Blue": [False, False, False, False, False, False, False],
    "Blue-White": [True, False, False, False, False, False, False],
    "Orange": [False, True, False, False, False, False, False],
    "Orange-Red": [False, False, True, False, False, False, False],
    "Red": [False, False, False, True, False, False, False],
    "White": [False, False, False, False, True, False, False],
    "Yellow": [False, False, False, False, False, True, False],
    "Yellow-White": [False, False, False, False, False, False, True]
}
spectral_classes = { 
    "A": [False, False, False, False, False, False],
    "B": [True, False, False, False, False, False],
    "F": [False, True, False, False, False, False],
    "G": [False, False, True, False, False, False],
    "K": [False, False, False, True, False, False],
    "M": [False, False, False, False, True, False],
    "O": [False, False, False, False, False, True]
}

@app.route("/api")
def test():
    return jsonify(time = str(datetime.now().time()))

@app.route("/api/identify", methods = ["POST"])
def identify():
    form = request.form
    X = [ form["temp"], form["lum"], form["rad"], form["mag"] ]
    X.extend(colors[form["color"]])
    X.extend(spectral_classes[form["spec"]])
    return jsonify({ "prediction": classes[model_tree.predict(X)[0]] })
