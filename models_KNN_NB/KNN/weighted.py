# Import necessary libraries
import os  # For working with file paths

import pandas as pd  # For data loading and manipulation
from sklearn.metrics import accuracy_score  # For computing classification accuracy
from sklearn.model_selection import cross_val_score, train_test_split  # For cross-validation and data splitting
from sklearn.neighbors import KNeighborsClassifier  # For KNN classification
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Load the data from the CSV file
weather_data = pd.read_csv(os.path.join("E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/nghean.csv"))

# Preprocess the data
# Drop columns that are not needed
weather_features = weather_data.drop(["name","datetime","description","icon"], axis=1)

# Fill in missing values with the column means
weather_features = weather_features.fillna(weather_features.mean())

# Scale numerical features
scaler = StandardScaler()
weather_features = scaler.fit_transform(weather_features)

# Define the output column of the data
weather_labels = weather_data["icon"]

# Split the data into training and testing sets
# test_size=0.2 specifies a 20% testing set and 80% training set
train_features, test_features, train_labels, test_labels = train_test_split(weather_features, weather_labels,
                                                                            test_size=0.2)

# Create the model
weighted_knn = KNeighborsClassifier(n_neighbors=10, weights='distance')

# Use cross-validation to estimate model performance
# cv=5 specifies 5-fold cross-validation
scores = cross_val_score(weighted_knn, weather_features, weather_labels, cv=5)

# Train the model on the training set
weighted_knn.fit(train_features, train_labels)

# Make predictions on the testing set
predicted_labels = weighted_knn.predict(test_features)

# Compute the accuracy score for the model on the testing set
test_accuracy = round(accuracy_score(test_labels, predicted_labels), 4) * 100

# Print the accuracy score for the model on the testing set
print(f'KNN Weighted K=10 Accuracy = {test_accuracy}')

from joblib import dump,load

# Lưu trữ mô hình Gaussian Naive Bayes
dump(weighted_knn, 'E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/weighted_knn_model.joblib')

# Lưu trữ StandardScaler
dump(scaler, 'E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/scaler.joblib')

