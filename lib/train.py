import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn import metrics
from sklearn import model_selection
import time
import pickle
import re


class DataModel:
    def __init__(self):
        self.sample_texts = []
        self.labels = []


def prepare(data):
    labels = [l for l in os.listdir(data) if os.path.isdir(os.path.join(data, l))]

    txt_corpus = DataModel()

    for i, label in enumerate(labels):
        sample_path = os.path.join(data, label)
        samples = [s for s in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, s))]
        stats_label = {}
        for sample_file in samples:
            sample_file = os.path.join(sample_path, sample_file)
            with open(sample_file, 'r') as reader:
                lines = reader.read().splitlines()
                for line in lines:
                    txt_corpus.sample_texts.append(no_number_preprocessor(line))
                    txt_corpus.labels.append(label)

                if label in stats_label:
                    stats_label[label] = stats_label[label] + len(lines)
                else:
                    stats_label[label] = len(lines)

        for key in stats_label:
            print(f"added {key} samples ({stats_label[key]})")

    print('labels: ', len(set(txt_corpus.labels)))
    print('samples: ', len(txt_corpus.sample_texts))

    return txt_corpus


def no_number_preprocessor(tokens):
    r = re.sub('(\d)+', '', tokens)
    # This alternative just removes numbers:
    # r = re.sub('(\d)+', 'NUM', tokens.lower())
    return r


def stratify_split_train_test(data_model: DataModel, test_size=0.40):
    x_train, x_test, y_train, y_test = train_test_split(data_model.sample_texts,
                                                        data_model.labels,
                                                        stratify=data_model.labels,
                                                        test_size=test_size)

    print('train size (x, y): ', len(x_train), len(y_train))
    print('test size (x, y): ', len(x_test), len(y_test))
    return x_train, x_test, y_train, y_test


def train(model: Pipeline, x_train, y_train):
    print("start fitting model")
    start = time.perf_counter()
    text_clf = model.fit(x_train, y_train)
    end = time.perf_counter()
    print(f"model fitting toke {end - start:0.4f} seconds")
    return text_clf


def create_model():
    vec_word = TfidfVectorizer(analyzer='word',
                               ngram_range=(1, 5),
                               min_df=5,
                               lowercase=True)

    vec_char_wb = TfidfVectorizer(analyzer='char_wb',
                                  ngram_range=(1, 5),
                                  min_df=5,
                                  lowercase=True)

    union = FeatureUnion([("vec_word", vec_word), ("vec_char_wb", vec_char_wb)])
    pipeline = Pipeline([('vect', union), ('clf', MultinomialNB(alpha=1.0))])

    print("create model definition", pipeline)
    return pipeline


def evaluate_train_set(classifier, x_train, y_train):
    folds = 5
    scoring_types = ['f1_weighted', 'accuracy']

    print(f"evaluate train set with {scoring_types} and {folds} folds")

    for scoring in scoring_types:
        scores = model_selection.cross_val_score(classifier, X=x_train, y=y_train,
                                                 cv=folds, scoring=scoring)
        sigma_2 = scores.std() * 2
        score_range = [round(scores.mean() - sigma_2, 4), round(scores.mean() + sigma_2, 4)]

        print(f"{scoring} scores: {scores}")
        print(f"{scoring} mean score: {scores.mean():0.4f}")
        print(f"{scoring} std {scores.std():0.4f}")
        print(f"{scoring} score range (+/- 2 * std): {score_range}")


def evaluate_test_set(classifier: Pipeline, x_test, y_test):
    predicted = classifier.predict(x_test)
    print(metrics.classification_report(
        y_test, predicted,
        digits=4))


def show_top10(pipeline, n=10):
    feature_names = np.asarray(pipeline['vect'].get_feature_names_out())
    print(f"top {n} features per label")
    for i, label in enumerate(pipeline['clf'].classes_):
        top_features = np.argsort(pipeline['clf'].feature_log_prob_[i])[-n:]
        # print(f"{label}: {[x.encode('unicode_escape') for x in feature_names[top_features]]}")
        print(f"{label}: {feature_names[top_features]}")

    print(f"number of features: {len(pipeline['vect'].get_feature_names_out())}")


def save_model(model):
    model_name = "model.pkl"
    print("saving model")
    pickle.dump(model, open(model_name, 'wb'))
    model_size = os.path.getsize(model_name)
    print(f'model size (Bytes): {model_size}')
    print(f'model size (KiB): {model_size / 1024:0.2f}')
    print(f'model size (MiB): {model_size / (1024 * 1024):0.2f}')
    print(f'model size (GiB): {model_size / (1024 * 1024 * 1024):0.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language classifier.')
    parser.add_argument("--data", type=str, default="data")
    args = parser.parse_args()

    txt_corpus = prepare(args.data)
    x_train, x_test, y_train, y_test = stratify_split_train_test(txt_corpus)
    model = create_model()

    evaluate_train_set(model, x_train, y_train)

    train(model, x_train, y_train)
    show_top10(model)
    evaluate_test_set(model, x_test, y_test)
    save_model(model)
