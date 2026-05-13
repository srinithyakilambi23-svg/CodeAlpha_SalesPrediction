import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Advertising.csv")

# Show column names
print(data.columns)

# Features
X = data[['TV', 'radio', 'newspaper']]

# Target
y = data['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Error
error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", error)

# Graph
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")
plt.show()