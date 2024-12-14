from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Create SVM model
svm_model = SVC(kernel='linear')  # You can try 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)


# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))