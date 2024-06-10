import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data
file_path = 'C:\\Users\\CHITTURI SRINIVAS\\Downloads\\heart.csv'
heart_data = pd.read_csv(file_path)

# Feature Engineering
heart_data['age_chol'] = heart_data['age'] * heart_data['chol']
heart_data['age_thalach'] = heart_data['age'] * heart_data['thalach']
heart_data['oldpeak_squared'] = heart_data['oldpeak'] ** 2

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

# Initialize and fit the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(features, target)

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
