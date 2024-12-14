import numpy as np
import random

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

# Load and preprocess the dataset
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')
X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

# Reshape and normalize the data
X_train = X_train.reshape(len(X_train), 100, 100, 3) / 255.0
X_test = X_test.reshape(len(X_test), 100, 100, 3) / 255.0

# Define CNN for feature extraction
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten()  # Flatten layer outputs the features
])

# Extract features using the CNN
X_train_features = cnn_model.predict(X_train)
X_test_features = cnn_model.predict(X_test)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # 3-nearest neighbors
knn.fit(X_train_features, Y_train)

# Predict and evaluate
Y_pred = knn.predict(X_test_features)
print("KNN Accuracy:", accuracy_score(Y_test, Y_pred))

# Predict a random test image
idx = random.randint(0, len(Y_test) - 1)
random_image = X_test[idx, :]
random_image_features = cnn_model.predict(random_image.reshape(1, 100, 100, 3))
prediction = knn.predict(random_image_features)

label = "Cat" if prediction[0] == 1 else "Dog"
plt.imshow(random_image)
plt.title(f"This is a {label}")
plt.show()
