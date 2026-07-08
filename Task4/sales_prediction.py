# CodSoft Data Science Internship
# Task 4 - Sales Prediction Using Python
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load Dataset
df = pd.read_csv("Advertising.csv")
# Display Dataset
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
# Data Visualization
# TV vs Sales
plt.figure(figsize=(8,6))
plt.scatter(df["TV"], df["Sales"])
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.grid(True)
plt.show()

# Radio vs Sales
plt.figure(figsize=(8,6))
plt.scatter(df["Radio"], df["Sales"])
plt.xlabel("Radio Advertising")
plt.ylabel("Sales")
plt.title("Radio Advertising vs Sales")
plt.grid(True)
plt.show()

# Newspaper vs Sales
plt.figure(figsize=(8,6))
plt.scatter(df["Newspaper"], df["Sales"])
plt.xlabel("Newspaper Advertising")
plt.ylabel("Sales")
plt.title("Newspaper Advertising vs Sales")
plt.grid(True)
plt.show()
# Prepare Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Predict New Sales

sample = pd.DataFrame(
    [[230.1, 37.8, 69.2]],
    columns=["TV", "Radio", "Newspaper"]
)

prediction = model.predict(sample)

print("\nPredicted Sales:")
print(prediction[0])