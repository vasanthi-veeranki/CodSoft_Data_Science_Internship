import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Convert Gender to Numbers
encoder = LabelEncoder()
df["Sex"] = encoder.fit_transform(df["Sex"])

# Select Features and Target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")

# User Input for Prediction
print("\nEnter Passenger Details")

pclass = int(input("Passenger Class (1/2/3): "))
gender = input("Gender (male/female): ").lower()
age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouses: "))
parch = int(input("Number of Parents/Children: "))
fare = float(input("Fare: "))

# Convert Gender
sex = 1 if gender == "male" else 0

# Create Input DataFrame
sample = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare]
})

# Predict
prediction = model.predict(sample)

# Output Result
if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did Not Survive")