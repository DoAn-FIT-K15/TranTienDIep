import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def predict_with_model(model, features):
    predictions = model.predict(features)
    return predictions

def new_fusion_model(X_train, y_train, X_test, n_neighbors_percent=20):
    n_neighbors = max(1, int(n_neighbors_percent / 100 * len(X_train)))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    neighbors = knn.kneighbors(X_test, return_distance=False)

    nb = GaussianNB()
    fusion_preds = []
    for i in range(len(X_test)):
        nb.fit(X_train.iloc[neighbors[i]], y_train.iloc[neighbors[i]])
        pred = nb.predict(X_test[i:i+1])
        fusion_preds.append(pred[0])
    return np.array(fusion_preds)
