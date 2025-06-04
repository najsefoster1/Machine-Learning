#Najse Foster
#Professor JC Martel
#CIS625 - Machine Learning for Business
#Unit 6: Assignment
#Neural Network Replication using Wine Quality Dataset


#Import all Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Load Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(url, sep=';')

#Preprocess Data
data['quality'] = (data['quality'] >= 7).astype(int)
X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Build and Train Model
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

#Evaluate Model
y_pred = model.predict(X_test_scaled)

#Print Train and Test Accuracy
print(f"Training Accuracy: {model.score(X_train_scaled, y_train):.4f}")
print(f"Testing Accuracy: {model.score(X_test_scaled, y_test):.4f}")

#Print Classification Report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)

#Save Report to Text File
with open('classification_report.txt', 'w') as f:
    f.write(report)

#Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()