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
X_train, X_test, y_train, y_test = train_test_split(weather_features, weather_labels, test_size=0.2, random_state=42)

# Định nghĩa mô hình Naive Bayes
nb = GaussianNB()

# Tạo danh sách để lưu kết quả
neighbor_percentages = [1, 5, 10, 15, 20, 40]  # Các tỷ lệ phần trăm của n_neighbors
fusion_accuracies = []

# Thực hiện thử nghiệm với các tỷ lệ phần trăm khác nhau
for percent in neighbor_percentages:
    n_neighbors = max(1, int((percent / 100) * len(X_train)))  # Tính số lân cận dựa trên phần trăm
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Định nghĩa và sử dụng mô hình fusion
    def fusion_model(X_train, y_train, X_test):
        knn.fit(X_train, y_train)
        neighbors = knn.kneighbors(X_test, return_distance=False)
        fusion_preds = []
        for i in range(len(X_test)):
            nb.fit(X_train.iloc[neighbors[i]], y_train.iloc[neighbors[i]])
            pred = nb.predict(X_test.iloc[i:i+1])
            fusion_preds.append(pred[0])
        return np.array(fusion_preds)

    # Huấn luyện và kiểm tra độ chính xác của mô hình fusion
    fusion_preds = fusion_model(X_train, y_train, X_test)
    accuracy = np.mean(fusion_preds == y_test) * 100
    fusion_accuracies.append(accuracy)

# Tạo DataFrame từ kết quả
results_df = pd.DataFrame({
    'Neighbor Percentage': neighbor_percentages,
    'Accuracy': fusion_accuracies
})

# Vẽ biểu đồ kết quả dạng cột
plt.figure(figsize=(10, 6))
plt.bar(neighbor_percentages, fusion_accuracies, color='skyblue')
plt.xlabel('% of training data as K')
plt.ylabel('Accuracy %')
plt.title('Combined KNN Bayes Evaluation')
plt.ylim(min(fusion_accuracies) - 5, 100)  # Điều chỉnh để biểu đồ trông đẹp hơn
plt.grid(True)
plt.show()

# In DataFrame
print(results_df)
