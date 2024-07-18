# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the merged data
diabetes_data = pd.read_csv('FinalWith_Filled.csv')

# Split data into features and target variable
X = diabetes_data.drop('Target', axis=1)
y = diabetes_data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Generate classification report
#report = classification_report(y_test, y_pred)
#print("Classification Report:\n", report)
# Creating a pickle file for the classifier
filename= 'diabetes-type-predictor1.pkl'
pickle.dump(model, open(filename, 'wb'))