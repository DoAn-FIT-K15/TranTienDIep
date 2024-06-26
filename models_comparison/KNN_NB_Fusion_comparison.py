# Imports necessary libraries
import os  # For working with file paths

import matplotlib.pyplot as plt  # For graph making
import numpy as np  # For mathematical
import pandas as pd  # For data loading and manipulation
from sklearn.model_selection import train_test_split  # For data splitting
from sklearn.naive_bayes import GaussianNB  # For NB classification
from sklearn.neighbors import KNeighborsClassifier  # For KNN classification

# Load and preprocess the data
weather_data = pd.read_csv(os.path.join("E:\DOAN\TranTienDiep_2020603359\TranTienDIep\data/nghean.csv"))
weather_features = weather_data.drop(["name","datetime","description","icon"], axis=1)
weather_features = weather_features.fillna(weather_features.mean())
weather_labels = weather_data["icon"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(weather_features, weather_labels, test_size=0.2)

# Define the models
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


# Compare the accuracy of the two models and the fusion model over 50 trials
num_trials = 5
estimate_nb_accuracy = []
uniform_knn_accuracy = []
fusion_accuracy = []

for i in range(num_trials):
    # Shuffle the training data
    idx = np.random.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train.iloc[idx], y_train.iloc[idx]

    # Fit the models on the shuffled training data
    nb.fit(X_train_shuffled, y_train_shuffled)
    knn.fit(X_train_shuffled, y_train_shuffled)
    fusion_preds = fusion_model(X_train, y_train, X_test, knn, nb)

    # Compute the accuracy on the test data
    estimate_nb_accuracy.append(nb.score(X_test, y_test))
    uniform_knn_accuracy.append(knn.score(X_test, y_test))
    fusion_accuracy.append(np.mean(fusion_preds == y_test))

# Plot the results
models = ['Estimate Prior', 'Uniform KNN', 'Fusion']
x_pos = np.arange(len(models))
means = [np.mean(estimate_nb_accuracy) * 100, np.mean(uniform_knn_accuracy) * 100, np.mean(fusion_accuracy) * 100]
stds = [np.std(estimate_nb_accuracy) * 100, np.std(uniform_knn_accuracy) * 100, np.std(fusion_accuracy) * 100]

# Create a bar plot with error bars for all models
fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10, width=0.5,
       color=['#1f77b4', '#ff7f0e', '#2ca02c']
)

# Set the graph properties
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('Accuracy % Over 50 Trials')
ax.set_ylim([0, 100])
ax.set_title('KNN VS NB VS Fusion')
ax.yaxis.grid(True)

# Add the accuracy percentage on top of each bar
for i, v in enumerate(means):
    ax.text(i, v + 1, str(round(v, 2)), color='black', ha='center')

# Show the plot
plt.show()
