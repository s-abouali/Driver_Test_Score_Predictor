import numpy as np
from sklearn.linear_model import LinearRegression

# Practice driving hours
practice_hours = np.array([5, 10, 15, 20, 25, 30, 35, 40]).reshape(-1, 1)

# Driving test scores
scores = np.array([50, 58, 65, 72, 78, 84, 90, 95])

# Train model
model = LinearRegression()
model.fit(practice_hours, scores)

# Predict
hours = float(input("Enter practice driving hours: "))
predicted_score = model.predict([[hours]])

print(f"Predicted driving test score: {predicted_score[0]:.1f}")