import pandas as pd
from Console.package import Functions
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Create an instance of the Functions class
f = Functions()

# Import data using get_args
df = f.get_args("df")
target_col = f.get_args("target_col")
# Specify the target variable column and other parameters
target_variable = target_col
threshold = 0.5
activation_param = "tanh"
solver_param = "sgd"
hidden_layer_sizes_param = (8, 8)

# Assuming your target variable is binary (0 or 1), you can convert it to categorical
df[target_col] = pd.Categorical(df[target_col]).codes

# Split the data into features (X) and target variable (y)
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the ANN model
model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes_param,
    activation=activation_param,
    solver=solver_param,
    random_state=42,
    max_iter=500
)

model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Check if predictions are above the specified threshold
predictions_above_threshold = (model.predict_proba(X_test_scaled)[:, 1] > threshold).astype(int)
print("Predictions above threshold:", predictions_above_threshold)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ["Actual 0", "Actual 1"]
tick_marks = range(len(classes))
plt.xticks(tick_marks, ["Predicted 0", "Predicted 1"])
plt.yticks(tick_marks, classes)

plt.ylabel('Actual')
plt.xlabel('Predicted')
f.save_image(plt)

model_filename = "ann_model.pkl"
joblib.dump(model, model_filename)
print(f"Trained model saved as {model_filename}")