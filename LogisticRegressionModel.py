import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib  # Import joblib for saving the model

# Load your dataset
dataset_path = '/content/SC data for Models.csv'
df = pd.read_csv(dataset_path)

# Extract features and target variable
X = df.drop('Gender', axis=1)  # Replace 'target_variable' with your actual target variable
y = df['Gender']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Get the coefficients from the logistic regression model
logistic_coefficients = logistic_model.coef_[0]

# Create a DataFrame with features and their coefficients
logistic_fi_df = pd.DataFrame({'Features': X.columns, 'Coefficient': logistic_coefficients})

# Sort features by coefficient values
logistic_fi_df = logistic_fi_df.sort_values(by='Coefficient', ascending=False)

# Display the feature importance table for Logistic Regression
print("Logistic Regression Coefficients:")
print(logistic_fi_df)

# Optionally, you can visualize the feature importances for Logistic Regression
plt.figure(figsize=(10, 6))
plt.bar(logistic_fi_df['Features'], logistic_fi_df['Coefficient'])
plt.xlabel('Features')
plt.ylabel('Coefficient (Importance)')
plt.title('Logistic Regression Coefficients (Feature Importance)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Save the model as a pickle file
model_filename = "logistic_model.pkl"  # Replace with your desired filename
joblib.dump(logistic_model, model_filename)
print(f"Logistic Regression model saved as: {model_filename}")
