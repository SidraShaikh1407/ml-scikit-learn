import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Create dataset
hours = np.array([1,2,3,4,5,6,7,8,9,10])
marks = np.array([35,40,50,55,65,70,75,80,88,95])

df = pd.DataFrame({"Hours": hours, "Marks": marks})

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(df[['Hours']], df['Marks'], test_size=0.3, random_state=42)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Check performance
print("R2 Score (Accuracy):", r2_score(y_test, y_pred))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))

# Extra: Predict for new student
new_hours = 6.5
predicted_marks = model.predict(pd.DataFrame({'Hours':[new_hours]}))
print(f"\nPredicted Marks for {new_hours} hours study: {predicted_marks[0]:.2f}")


# Step 6: Visualization
plt.scatter(hours, marks)
plt.plot(hours, model.predict(df[['Hours']]))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

