
import pandas as pd

# Load the dataset
data = pd.read_csv (r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')

# Display the first few rows of the dataset
print(data.head())

# Display summary information about the dataset
print(data.info())

# Display basic statistics of the dataset
print(data.describe(include='all'))

# Display the shape of the dataset (number of rows and columns)
print("Shape of the dataset:", data.shape)

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv( r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')

# Pie chart for movie type distribution
type_counts = data['type'].value_counts()
plt.figure(figsize=(8, 8))
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Movie Types')
plt.ylabel('')  # Hide the y-label
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Time Series Analysis: Number of movies released each year
yearly_data = data['release_year'].value_counts().sort_index()
yearly_data.index = pd.to_datetime(yearly_data.index, format='%Y')

# Ensure the index has a frequency and fill missing values
yearly_data = yearly_data.asfreq('YE').fillna(0)
yearly_data[yearly_data <= 0] = 1e-6

# Decompose the time series
decomposition = seasonal_decompose(yearly_data, model='multiplicative')
decomposition.plot()
plt.show()

# Exponential Smoothing (Holt-Winters)
model_hw = ExponentialSmoothing(yearly_data, seasonal='multiplicative', seasonal_periods=10).fit()
hw_forecast = model_hw.forecast(5)

# ARIMA model
model_arima = ARIMA(yearly_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 10)).fit()
arima_forecast = model_arima.forecast(5)

# Plot forecasts
plt.figure(figsize=(10, 6))
plt.plot(yearly_data.index, yearly_data, label='Original')
plt.plot(hw_forecast.index, hw_forecast, label='Holt-Winters Forecast')
plt.plot(arima_forecast.index, arima_forecast, label='ARIMA Forecast')
plt.legend()
plt.show()


from textblob import TextBlob

# Sentiment Analysis on descriptions
data['sentiment'] = data['description'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)

# Visualize sentiment
data['sentiment'].plot(kind='hist', bins=50, title='Sentiment Distribution')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')

# Check the structure of the dataset
print(data.info())

# Convert 'rating' to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Drop rows with missing values in 'rating'
data = data.dropna(subset=['rating'])

# Plot histogram of rating distribution
plt.figure(figsize=(10, 6))
plt.hist(data['rating'], bins=50, color='skyblue', edgecolor='black')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv(r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')

# Convert 'duration' to numeric (assuming it contains mixed types)
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')

# Fill missing values in 'duration' with the mean
data['duration'] = data['duration'].fillna(data['duration'].mean())

# Convert 'rating' to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Fill missing values in 'rating' with the mean
data['rating'] = data['rating'].fillna(data['rating'].mean())

# Drop rows with any remaining NaN values
data.dropna(inplace=True)

# Check if there are any valid samples left
if data.shape[0] == 0:
    print("No valid samples remaining after preprocessing.")
else:
    # Prepare data for clustering
    X = data[['duration', 'rating']].values

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', label='Data Points')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Cluster Centers')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Rating')
    plt.title('K-Means Clustering of Movies')
    plt.legend()
    plt.show()

    print("Cluster centers:", kmeans.cluster_centers_)
    print("Labels:", kmeans.labels_)
data.head()



import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your data
data = pd.read_csv(r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')  # Replace with the path to your actual data file

# Function to convert duration from 'XX min' to numeric
def convert_duration(duration_str):
    try:
        return float(duration_str.replace(' min', ''))
    except ValueError:
        return None

# Apply the conversion function to the duration column
data['duration'] = data['duration'].apply(convert_duration)

# Drop rows with invalid durations
data = data.dropna(subset=['duration', 'rating'])

# Convert rating to numeric and fill invalid entries with 0
data['rating'] = pd.to_numeric(data['rating'], errors='coerce').fillna(0)

# Ensure all ratings are floats
data['rating'] = data['rating'].astype(float)

# Prepare the data for clustering
X = data[['duration', 'rating']].values

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.xlabel('Duration (minutes)')
plt.ylabel('Rating')
plt.title('K-Means Clustering of Movies')
plt.show()

print("Process finished successfully")



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Topic Modeling on descriptions
vectorizer = CountVectorizer(stop_words='english')
text_vectorized = vectorizer.fit_transform(data['description'].dropna())
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(text_vectorized)

# Display topics
for i, topic in enumerate(lda.components_):
    print(f"Topic {i}:")
    print([vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]])
data.head()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv')

# Convert 'duration' to numeric (assuming it contains mixed types)
data['duration'] = pd.to_numeric(data['duration'], errors='coerce')

# Fill missing values in 'duration' with the mean
#data['duration'].fillna(data['duration'].mean(), inplace=True)
data['duration'] = data['duration'].fillna(data['duration'].mean())
# Convert 'rating' to numeric
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')

# Fill missing values in 'rating' with the mean
#data['rating'].fillna(data['rating'].mean(), inplace=True)
data['rating'] = data['rating'].fillna(data['rating'].mean())

# Check for and handle cases where all values might be NaN after conversion
if data['duration'].isnull().all() or data['rating'].isnull().all():
    print("All values in 'duration' or 'rating' are NaN after conversion. Cannot proceed.")
else:
    # Drop rows with any remaining NaN values
    data.dropna(inplace=True)

    # Prepare data for classification
    X = data[['duration', 'rating']].values
    y = data['type'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Plotting bar graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Bar graph for actual labels
    unique_labels, counts_actual = pd.unique(y_test, return_counts=True)
    axes[0].bar(unique_labels, counts_actual, color='skyblue')
    axes[0].set_xlabel('Actual Labels')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Actual Labels Distribution')

    # Bar graph for predicted labels
    unique_labels, counts_predicted = pd.unique(y_pred, return_counts=True)
    axes[1].bar(unique_labels, counts_predicted, color='salmon')
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Predicted Labels Distribution')

    plt.tight_layout()
    plt.show()

    print("Accuracy:", accuracy)
    print("Classification Report:\n", class_report)



import pandas as pd

# Load the dataset
file_path = r'C:\Users\CHITTURI SRINIVAS\Downloads\disney_plus_titles.csv'

data= pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(data.head())

import pandas as pd

# Load the dataset


# Display basic information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Save a summary to a CSV file
summary = data.describe(include='all')
#summary.to_csv('/path/to/save/summary.csv')

# Example: Analyze the distribution of content ratings
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='rating', order=data['rating'].value_counts().index)
plt.title('Distribution of Content Ratings')
plt.xticks(rotation=45)
plt.show()

from textblob import TextBlob

# Example: Perform sentiment analysis on title descriptions
data['description'] = data['description'].fillna('')

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

data['sentiment'] = data['description'].apply(get_sentiment)

# Plot sentiment distribution
sns.histplot(data=data, x='sentiment', bins=20, kde=True)
plt.title('Sentiment Analysis of Descriptions')
plt.show()

# Advanced Analysis Example

# Implement time series analysis for release_year
release_years = data['release_year'].value_counts().sort_index()
release_years.plot(kind='line', figsize=(10, 6))
plt.title('Number of Titles Released Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.show()

# Perform clustering on genres
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Fill missing genres with empty strings
data['genre'] = data['genre'].fillna('')

# Vectorize genres
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['genre'])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

# Add cluster labels to the original data
data['cluster'] = kmeans.labels_

# Visualize clustering results
sns.countplot(data=data, x='cluster')
plt.title('Number of Titles per Cluster')
plt.show()

# Save the clustered data to a new CSV file
#data.to_csv('/path/to/save/clustered_data.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import os

# Load the dataset
file_path = 'C:/Users/CHITTURI SRINIVAS/PycharmProjects/pythonintern/disney_plus_titles.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())
print(data.describe())
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Ensure the output directory exists
output_dir = 'C:/Users/CHITTURI SRINIVAS/PycharmProjects/pythonintern/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save a summary to a CSV file
summary = data.describe(include='all')
summary.to_csv(os.path.join(output_dir, 'summary.csv'))

# Example: Analyze the distribution of content ratings
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='rating', order=data['rating'].value_counts().index)
plt.title('Distribution of Content Ratings')
plt.xticks(rotation=45)
plt.show






