#ml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset (Area in SqFt, Price in Lakhs)
area = np.array([500, 600, 700, 800, 900, 1000]).reshape(-1, 1)
price = np.array([30, 35, 40, 50, 55, 60])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(area, price, test_size=0.2, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(model.predict([[1200]]))
