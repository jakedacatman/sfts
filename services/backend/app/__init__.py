from flask import Flask, jsonify, request
from datetime import datetime
from tensorflow.keras.models import load_model 
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
prefix = "models/"

with open(f"{prefix}decision_tree.pkl", "rb") as file:
    model_tree = pickle.load(file)

model_mlp = load_model(f"{prefix}classification_mlp.keras")

with open(f"{prefix}mlp_processor.pkl", "rb") as file:
    proc_mlp = pickle.load(file)

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
    "Blue": [0, 0, 0, 0, 0, 0, 0],
    "Blue-White": [1, 0, 0, 0, 0, 0, 0],
    "Orange": [0, 1, 0, 0, 0, 0, 0],
    "Orange-Red": [0, 0, 1, 0, 0, 0, 0],
    "Red": [0, 0, 0, 1, 0, 0, 0],
    "White": [0, 0, 0, 0, 1, 0, 0],
    "Yellow": [0, 0, 0, 0, 0, 1, 0],
    "Yellow-White": [0, 0, 0, 0, 0, 0, 1]
} 
spectral_classes = {
    "A": [0, 0, 0, 0, 0, 0],
    "B": [1, 0, 0, 0, 0, 0],
    "F": [0, 1, 0, 0, 0, 0],
    "G": [0, 0, 1, 0, 0, 0],
    "K": [0, 0, 0, 1, 0, 0],
    "M": [0, 0, 0, 0, 1, 0],
    "O": [0, 0, 0, 0, 0, 1]
} 

@app.route("/api")
def test():
    return jsonify(time = str(datetime.now().time()))

@app.route("/api/identify", methods = ["POST"])
def identify():
    form = request.form
    X = [ form["temperature"], form["luminosity"], form["radius"], form["magnitude"] ]
    result = None
    if form["mlp"] == False or form["mlp"] == "":    
        X.extend(colors[form["color"]])
        X.extend(spectral_classes[form["class"]])
        X = np.array(X).reshape(1, -1)
        #print(X, X.shape)
        result = classes[model_tree.predict(X)[0]]
    else:
        X.extend([form["color"]])
        X.append(form["class"])
        X = np.array(X).reshape(1, -1)
        #print(X, X.shape)
        X = pd.DataFrame(
                data = X,
                index = [ 0 ],
                columns = [ 
                    "Temperature (K)",
                    "Luminosity(L/Lo)",
                    "Radius(R/Ro)",
                    "Absolute magnitude(Mv)",
                    "Star color",
                    "Spectral Class" 
                ]
        )

        X = proc_mlp.transform(X)
        result = classes[np.argmax(model_mlp.predict(X), axis = 1)[0]]

    return jsonify({ "prediction": result })
