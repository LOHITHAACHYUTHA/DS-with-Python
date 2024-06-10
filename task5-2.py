import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



# Load the data
import pandas as pd
# Load the dataset
file_path = 'C:\\Users\\CHITTURI SRINIVAS\\Downloads\\heart.csv'
data = pd.read_csv(file_path)
# Display the first few rows and the info of the dataset
print(data.head())
#print(data.info())

heart_data = pd.read_csv(file_path)
import matplotlib.pyplot as plt
data.hist(bins=50,grid=True,figsize=(10,10))
plt.show();



# Feature Engineering
# 1. Interaction features
heart_data['age_chol'] = heart_data['age'] * heart_data['chol']
heart_data['age_thalach'] = heart_data['age'] * heart_data['thalach']
heart_data['age_oldpeak'] = heart_data['age'] * heart_data['oldpeak']

# 2. Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
poly_features = poly.fit_transform(heart_data[['age', 'chol', 'thalach', 'oldpeak']])
#poly_feature_names = poly.get_feature_names(['age', 'chol', 'thalach', 'oldpeak'])
poly_feature_names = poly.get_feature_names_out(['age', 'chol', 'thalach', 'oldpeak'])

poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
heart_data = pd.concat([heart_data, poly_df], axis=1)

# 3. Aggregation features (example: groupby statistics)
# Since the data is not suitable for groupby operations, this is just an example.
# heart_data['age_mean'] = heart_data.groupby('sex')['age'].transform('mean')

# Drop original columns that have been expanded
heart_data = heart_data.drop(columns=['age', 'chol', 'thalach', 'oldpeak'])





print("Display the first few rows of the dataset with the new features")

print(heart_data.head(10))






# Separate features and target
features = heart_data.drop(columns=['target'])
target = heart_data['target']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
pca.fit(features_scaled)
explained_variance = pca.explained_variance_ratio_

# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest model with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2',None],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and model
best_rf_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Predict on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)


# Initialize and fit the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features, target)

# Evaluate model performance on the original feature set
# Train a RandomForestClassifier on the original feature set
model_original = RandomForestClassifier(n_estimators=100, random_state=42)
model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print("\nAccuracy with original feature set:", accuracy_original*100)


# Get feature importances
importances = rf_model.feature_importances_
feature_names = features.columns

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['feature'], feature_importances_df['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest Classifier')
plt.gca().invert_yaxis()
plt.show()

# Display top features
print("Top features based on Random Forest importance:")
print(feature_importances_df.head(10))


# If needed, save the feature importances to a CSV file
feature_importances_df.to_csv('feature_importances.csv', index=False)


