import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load và tiền xử lý dữ liệu
weather_data = pd.read_csv(os.path.join("E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/nghean.csv"))
weather_features = weather_data.drop(["name", "datetime", "description", "icon"], axis=1)
weather_features = weather_features.fillna(weather_features.mean())
weather_labels = weather_data["icon"]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(weather_features, weather_labels, test_size=0.2)

# Định nghĩa mô hình Naive Bayes
nb = GaussianNB()
# Thiết lập số lân cận dựa trên 20% tập huấn luyện
n_neighbors = int(0.2 * len(X_train))  # 20% của tập huấn luyện
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Định nghĩa và sử dụng mô hình fusion
def fusion_model(X_train, y_train, X_test, knn, nb):
    knn.fit(X_train, y_train)
    neighbors = knn.kneighbors(X_test, return_distance=False)
    fusion_preds = []
    for i in range(len(X_test)):
        nb.fit(X_train.iloc[neighbors[i]], y_train.iloc[neighbors[i]])
        pred = nb.predict(X_test.iloc[i:i+1])
        fusion_preds.append(pred[0])
    return np.array(fusion_preds)

# Huấn luyện và kiểm tra độ chính xác của mô hình fusion
fusion_preds = fusion_model(X_train, y_train, X_test, knn, nb)
accuracy = np.mean(fusion_preds == y_test) * 100

# In ra độ chính xác
print(f'Accuracy of Fusion Model with 20% Neighbors: {accuracy:.2f}%')

