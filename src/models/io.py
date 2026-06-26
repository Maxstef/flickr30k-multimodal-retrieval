import joblib


def save_sklearn_model(model, path, compress=3):
    joblib.dump(model, path, compress=compress)


def load_sklearn_model(path):
    return joblib.load(path)
