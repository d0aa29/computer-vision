from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib


# Set the directory paths for training and testing images
train_dir = "D:/Alzheimer_s Dataset/Alzheimer_s Dataset/train"
test_dir = "D:/Alzheimer_s Dataset/Alzheimer_s Dataset/test"

# Set the input shape
input_shape = (128, 128, 3)
batch_size = 32

# Use ImageDataGenerator for data augmentation
data_generator = ImageDataGenerator(rescale=1./255)

# Load the training images
train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Load the test images
test_generator = data_generator.flow_from_directory(
    test_dir,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Define the CNN model architecture up to the feature extraction layer
model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5)
])

# Extract features from training images
train_features = model.predict(train_generator)
train_features = train_features.reshape(train_features.shape[0], -1)  # Flatten features for KNN
train_labels = train_generator.classes


# Extract features from test images
test_features = model.predict(test_generator)
test_features = test_features.reshape(test_features.shape[0], -1)
test_labels = test_generator.classes


# Use CatBoost classifier
cb_classifier = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=0)
cb_classifier.fit(train_features, train_labels)

# Make predictions on the test data
test_predictions = cb_classifier.predict(test_features)

# Calculate accuracy
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Save the Keras CNN model as an .h5 file
model.save('cnn_feature_extractor2.h5')

# Save the trained SVM model
joblib.dump(cb_classifier, 'cat_classifier.pkl')

#model.save('cnn_Feature_Extractor.keras')



from keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved models
cnn_model = load_model("C:/Users/doaaz/PycharmProjects/CV/sec5/cnn_Feature_Extractor2.h5")
cat_classifier = joblib.load("C:/Users/doaaz/PycharmProjects/CV/sec5/cat_classifier.pkl")


def classify_image(img_path):
    # Load and preprocess the new image
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to match the CNN input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to match the training preprocessing

    # Extract features using the CNN model
    features = cnn_model.predict(img_array)
    features = features.reshape(1, -1)  # Flatten to 1D for SVM

    # Classify the features with the catboost model
    prediction = cat_classifier.predict(features)

    # Interpret the prediction
    class_labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented',
                    3: 'VeryMildDemented'}  # Adjust based on your dataset labels
    result = class_labels[prediction[0]]
    return result


# Test the function with a new image
img_path = r"D:/Alzheimer_s Dataset/Alzheimer_s Dataset/test/NonDemented/32 (78).jpg"
result = classify_image(img_path)
print(f"The image is classified as: {result}")