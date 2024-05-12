import pandas as pd
from joblib import load

def load_scaler():
    scaler = load("E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/scaler.joblib")
    return scaler

def load_models():
    nb_model = load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/gaussian_naive_bayes_model.joblib')
    knn_model = load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/weighted_knn_model.joblib')
    return nb_model, knn_model

def load_training_data():
    X_train = pd.DataFrame(load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/X_train.joblib'))
    y_train = pd.Series(load('E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/y_train.joblib'))
    return X_train, y_train
