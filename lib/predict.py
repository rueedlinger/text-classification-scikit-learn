import pickle
import time


def load_model():
    with open('model.pkl', 'rb') as f:
        print("loading model")
        start = time.perf_counter()
        model = pickle.load(f)
        end = time.perf_counter()
        print(f"loading model toke {end - start:0.4f} seconds")
        return model


def predict(model):
    txt = input("Please enter sentence: ")

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


if __name__ == "__main__":
    model = load_model()
    predict(model)
