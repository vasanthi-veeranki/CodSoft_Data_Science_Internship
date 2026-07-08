# Task 3: Iris Flower Classification
# CodSoft Data Science Internship

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("IRIS.csv")

# Display Dataset
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Data Visualization
# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(df["sepal_length"], df["petal_length"])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.grid(True)
plt.show()

# Histogram
df.hist(figsize=(10,8))
plt.suptitle("Distribution of Iris Features")
plt.show()

# Bar Chart
plt.figure(figsize=(6,5))
df["species"].value_counts().plot(kind="bar")
plt.title("Species Count")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# Prepare Data
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict New Flower
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

prediction = model.predict(sample)

print("\nPredicted Species:")
print(prediction[0])