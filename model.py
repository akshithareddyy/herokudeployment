import numpy as np
import pandas as pd
import pickle

# Load the dataset
dataset = pd.read_csv("C:\\Users\\GOD\\Downloads\\hiringdata\\hiring.csv")

# Fill missing values in 'experience' with 0 and 'test_score(out of 10)' with the mean
dataset['experience'].fillna(0, inplace=True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(), inplace=True)

# Extract features (X) and target variable (y)
X = dataset[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = dataset['salary($)']

# Apply the function to convert 'experience' values to integers
X['experience'] = X['experience'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else 0)

# Train a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

# Load the model from the file
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Use a NumPy array for prediction
prediction = loaded_model.predict(np.array([[2, 9, 6]]))
print(prediction)

