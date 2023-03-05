import pickle
import time
import requests
import numpy as np


def load_model():
    with open('model.pkl', 'rb') as f:
        print("loading model")
        start = time.perf_counter()
        model = pickle.load(f)
        end = time.perf_counter()
        print(f"loading model toke {end - start:0.4f} seconds")
        return model


def predict(model):
    lang = input("Please select language: ")
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    txt = ""
    while len(txt) < 30:
        txt = requests.get(url).json()['extract']

    start = time.perf_counter()
    print("---")
    print(txt)
    print("---")
    print(model.predict([txt]))
    probs = model.predict_proba([txt])[0]
    classes = model.classes_
    values = []
    for i, p in enumerate(probs):
        values.append((classes[i], p))

    values = sorted(values, key=lambda x: x[1], reverse=True)
    for i in range(0, 3):
        print(values[i])

    print("---")
    end = time.perf_counter()
    print(f"prediction toke {end - start:0.4f} seconds")


def show_features(pipeline):
    print(pipeline)
    for f in pipeline['vect'].vocabulary_:
        print(f)

    """
    feature_names = np.asarray(pipeline['vect'].get_feature_names_out())
    print(f"top {n} features per label")
    for i, label in enumerate(pipeline['clf'].classes_):
        top_features = np.argsort(pipeline['clf'].feature_log_prob_[i])[-n:]
        #print(f"{label}: {[x.encode('unicode_escape') for x in feature_names[top_features]]}")
        print(f"{label}: {feature_names[top_features]}")
    """

if __name__ == "__main__":
    model = load_model()
    show_features(model)
