# -*- coding: utf-8 -*-
# Basic imports
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from surprise import Reader, Dataset, SVD as sp_SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score as sk_precision_score, recall_score as sk_recall_score, f1_score as sk_f1_score
from sklearn import model_selection as sk_ms, ensemble as sk_ensemble, metrics as sk_metrics
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
#from concurrent.futures import ThreadPoolExecutor


# Load Data
def load_data(filepath):
    df = pd.read_csv(filepath).sample(frac=0.1)  # Take only 10% of data for fast execution
    return df

df = load_data("/Users/maha/Downloads/train.csv")


## Data types
#df.info(null_counts = True)

Q1 = df['release_date'].quantile(0.25)
Q3 = df['release_date'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['release_date'] < lower_bound) | (df['release_date'] > upper_bound)]

# Create a box plot
plt.figure(figsize=(10, 6))
plt.boxplot(df['release_date'], vert=False, labels=['release_date'])
plt.scatter(outliers['release_date'], [1] * len(outliers), color='red', label='Outliers')
plt.title('Release Date Box Plot with Outliers')
plt.legend()
plt.show()

# Outlier
print(df['release_date'].max()) # 30000101
# Locate the final instance preceding 30000101 and substitute the most recent "plausible" date preceding this anomalous data entry.
print(sorted(list(df['release_date'].unique()))[-2]) # 20170313

# Number of unique songs
print(f'There are {df.media_id.nunique()} different songs in the dataset.')
df['media_id'] = df['media_id'].astype("category")

# Number of unique genres
print(f'There are {df.genre_id.nunique()} different genres in the dataset.')

# Convert genre_id a categorical variable
df['genre_id'] = df['genre_id'].astype("category")

print(f'There are {df.user_gender.nunique()} different user genders in the dataset.')

# Convert user_gender a categorical variable
df['user_gender'] = df['user_gender'].astype("category")

# Number of unique albums
print(f'There are {df.album_id.nunique()} different albums in the dataset.')
df['album_id'] = df['album_id'].astype("category")

# Number of unique platforms
print(f'There are {df.platform_name.nunique()} different platforms in the dataset.')

# Convert platform_name a categorical variable
df['platform_name'] = df['platform_name'].astype("category")

print(f'There are {df.listen_type.nunique()} different listen types in the dataset.')


# Convert listen_type a categorical variable
df['listen_type'] = df['listen_type'].astype("category")

print(f'There are {df.artist_id.nunique()} different artists in the dataset.')

## Make artist_id a categorical variable
df['artist_id'] = df['artist_id'].astype("category")

def aggregate_user_age(df):
    usr_age_agg = df[['user_id', 'user_age']].groupby('user_age').nunique().reset_index()
    return usr_age_agg

def plot_user_age_distribution(user_age_data):
    plt.figure(figsize=(10,5))
    gph = sns.barplot(x='user_age', y='user_id', data=user_age_data, color='blue')

    plt.xlabel("Age", size=16)
    plt.ylabel("Number of Users", size=16)
    plt.title("User Age Distribution", size=18)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    for bar in gph.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 size=10, color='black')

    plt.show()

# Usage
usr_age_data = aggregate_user_age(df)
plot_user_age_distribution(usr_age_data)

def aggregate_user_gender(df):
    usr_gender_agg = df[['user_id', 'user_gender']].groupby('user_gender').nunique().reset_index()
    return usr_gender_agg

def plot_user_gender_distribution(user_gender_data):
    plt.figure(figsize=(10,5))
    gph = sns.barplot(x='user_gender', y='user_id', data=user_gender_data, color='blue')

    plt.xlabel("Gender", size=16)
    plt.ylabel("Number of Users", size=16)
    plt.title("User Gender Distribution", size=18)

    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    gph.set_xticklabels(('Male', 'Female'))  # Change labels (0, 1 -> Man, Woman)

    for bar in gph.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 size=10, color='black')

    plt.show()

# Usage
usr_gender_data = aggregate_user_gender(df)
plot_user_gender_distribution(usr_gender_data)

def plot_weekday_user_count(df):
    # Select and aggregate data
    user_counts_by_weekday = df[['day_of_week', 'user_id']].groupby('day_of_week').nunique().reset_index()

    # Plot results
    plt.figure(figsize=(10,5))
    gph = sns.barplot(x='day_of_week', y='user_id', data=user_counts_by_weekday, color='blue')

    # Add labels
    plt.xlabel("Day of Week", size=16)
    plt.ylabel("Number of Users", size=16)

    # Disable Y-axis
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)

    # Change labels (0,1,2,3,4,5,6 -> Monday, Tuesday, ...)
    gph.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Add total per bar
    for bar in gph.patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{int(bar.get_height()):,}', ha='center', va='bottom',
                 size=10, color='black', weight='bold')

    plt.show()

# Usage
df['ts_listen'] = pd.to_datetime(df['ts_listen'], unit='s')
df['day_of_week'] = df['ts_listen'].dt.dayofweek

plot_weekday_user_count(df)

def plot_hourly_user_count(data):
    # Extract and categorize hour listened
    data['hour_listened'] = data['ts_listen'].dt.hour
    data['hour_listened'] = data['hour_listened'].astype('category')

    # Select and aggregate data
    hourly_user_counts = data[['hour_listened', 'user_id']].groupby('hour_listened').count().reset_index()

    # Plot results
    plt.figure(figsize=(10,5))
    gph = sns.barplot(x='hour_listened', y='user_id', data=hourly_user_counts, color='blue')

    # Add labels
    plt.xlabel("Hour", size=16)
    plt.ylabel("Number of Users", size=16)

    plt.show()

# Assuming df is your DataFrame with 'ts_listen' column
df['ts_listen'] = pd.to_datetime(df['ts_listen'], unit='s')

plot_hourly_user_count(df)

# Copy the original DataFrame to a new one
new_df = df.copy()

# Convert timestamp to datetime
new_df['ts_listen'] = pd.to_datetime(new_df['ts_listen'], unit='s')

# Age Grouping
bins = [0, 18, 25, 35, 50, 100]
labels = ['<18', '18-24', '25-34', '35-49', '50+']
new_df['age_group'] = pd.cut(new_df['user_age'], bins=bins, labels=labels, right=False)

# Artist Popularity
artist_popularity = new_df['artist_id'].value_counts().reset_index()
artist_popularity.columns = ['artist_id', 'artist_popularity']
new_df = new_df.merge(artist_popularity, on='artist_id', how='left')

# Album Popularity
album_popularity = new_df['album_id'].value_counts().reset_index()
album_popularity.columns = ['album_id', 'album_popularity']
new_df = new_df.merge(album_popularity, on='album_id', how='left')

# Genre Popularity
genre_popularity = new_df['genre_id'].value_counts().reset_index()
genre_popularity.columns = ['genre_id', 'genre_popularity']
new_df = new_df.merge(genre_popularity, on='genre_id', how='left')

# User Listening Frequency
user_listen_frequency = new_df['user_id'].value_counts().reset_index()
user_listen_frequency.columns = ['user_id', 'user_listen_frequency']
new_df = new_df.merge(user_listen_frequency, on='user_id', how='left')

# Release Year
new_df['release_year'] = new_df['release_date'].apply(lambda x: int(str(x)[:4]))

# Favorite Artist
favorite_artist = new_df.groupby('user_id')['artist_id'].apply(lambda x: x.value_counts().index[0]).reset_index()
favorite_artist.columns = ['user_id', 'favorite_artist']
new_df = new_df.merge(favorite_artist, on='user_id', how='left')

# Recent Trend
recent_trend = new_df.sort_values('ts_listen').groupby('user_id')['genre_id'].last().reset_index()
recent_trend.columns = ['user_id', 'recent_trend']
new_df = new_df.merge(recent_trend, on='user_id', how='left')

# Season of Release
new_df['release_month'] = new_df['release_date'].apply(lambda x: int(str(x)[4:6]))
seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
            5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
new_df['season_of_release'] = new_df['release_month'].map(seasons)

# Album Loyalty
album_loyalty = new_df.groupby(['user_id', 'album_id'])['media_id'].count().reset_index()
album_loyalty = album_loyalty.groupby('user_id')['media_id'].mean().reset_index()
album_loyalty.columns = ['user_id', 'album_loyalty']
new_df = new_df.merge(album_loyalty, on='user_id', how='left')

# Artist Loyalty
artist_loyalty = new_df.groupby(['user_id', 'artist_id'])['media_id'].count().reset_index()
artist_loyalty = artist_loyalty.groupby('user_id')['media_id'].mean().reset_index()
artist_loyalty.columns = ['user_id', 'artist_loyalty']
new_df = new_df.merge(artist_loyalty, on='user_id', how='left')

# Assuming columns artist_popularity and album_popularity exist
new_df['artist_album_interaction'] = new_df['artist_id'].astype(int) * new_df['album_id'].astype(int)

# Adding temporal features
new_df['ts_listen'] = pd.to_datetime(new_df['ts_listen'], unit='s')
new_df['day_of_week'] = new_df['ts_listen'].dt.dayofweek

# Omitting the first recommended track for each user
grouped = new_df.sort_values('ts_listen').groupby('user_id')
omitted_tracks = grouped.apply(lambda x: x.iloc[0]).reset_index(drop=True)
new_df = grouped.apply(lambda x: x.iloc[1:]).reset_index(drop=True)
new_df


# Collaborative Filtering (using Surprise library)

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(new_df[['user_id', 'media_id', 'is_listened']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

# Get the mean of the 'is_listened' variable
baseline_prediction = trainset.global_mean

# Create a list of baseline predictions (all will be the same)
baseline_predictions = [baseline_prediction] * len(testset)

# Assuming 'actual_labels_knn' is available
actual_labels_knn = [1, 0, 1, 0, 1]  # Replace with your actual labels

# Create a list of baseline predictions for these 5 samples
baseline_predictions = [baseline_prediction] * len(actual_labels_knn)

# Assuming binary predictions (you can adjust threshold if needed)
predicted_labels_baseline = [1 if x >= 0.5 else 0 for x in baseline_predictions]

# Calculate RMSE
rmse_baseline = np.sqrt(((np.array(actual_labels_knn) - np.array(baseline_predictions[:len(actual_labels_knn)]))**2).mean())

# Calculate Precision and Recall
precision_baseline = sk_precision_score(actual_labels_knn, predicted_labels_baseline)
recall_baseline = sk_recall_score(actual_labels_knn, predicted_labels_baseline)

# Calculate F1 Score
f1_baseline = sk_f1_score(actual_labels_knn, predicted_labels_baseline)

# Print the results
print(f"Baseline - RMSE: {rmse_baseline}, Precision: {precision_baseline}, Recall: {recall_baseline}, F1 Score: {f1_baseline}")

# SVD with hyperparameter tuning

param_grid_svd = {
    'n_factors': [50, 100, 150],
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
}

gs_svd = GridSearchCV(sp_SVD, param_grid_svd, measures=['rmse'], cv=3,
                      n_jobs=-1, refit=True, joblib_verbose=0)
gs_svd.fit(data)

algo_svd = gs_svd.best_estimator['rmse']
algo_svd.fit(trainset)
predictions_svd = algo_svd.test(testset)

# Evaluation SVD

# Calculate RMSE
rmse_svd = accuracy.rmse(predictions_svd)

# Calculate Precision and Recall
actual_labels = [1 if x.r_ui == 1 else 0 for x in predictions_svd]
predicted_labels = [1 if x.est >= 0.5 else 0 for x in predictions_svd]
precision_svd = sk_precision_score(actual_labels, predicted_labels)
recall_svd = sk_recall_score(actual_labels, predicted_labels)

# Calculate F1 Score
f1_svd = sk_f1_score(actual_labels, predicted_labels)

# Print the results
print(f"SVD - RMSE: {rmse_svd}, Precision: {precision_svd}, "
      f"Recall: {recall_svd}, F1 Score: {f1_svd}")

# KNN with hyperparameter tuning
param_grid_knn = {'k': [20, 40], 'sim_options': {'name': ['msd']}}
gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=3, n_jobs=-1)
gs_knn.fit(data)
algo_knn = gs_knn.best_estimator['rmse']
algo_knn.fit(trainset)
predictions_knn = algo_knn.test(testset)



# Calculate RMSE
rmse_knn = accuracy.rmse(predictions_knn)

# Calculate Precision and Recall
actual_labels_knn = [1 if x.r_ui == 1 else 0 for x in predictions_knn]
predicted_labels_knn = [1 if x.est >= 0.5 else 0 for x in predictions_knn]
precision_knn = sk_precision_score(actual_labels_knn, predicted_labels_knn)
recall_knn = sk_recall_score(actual_labels_knn, predicted_labels_knn)

# Calculate F1 Score
f1_knn = sk_f1_score(actual_labels_knn, predicted_labels_knn)

# Print the results
print(f"KNN - RMSE: {rmse_knn}, Precision: {precision_knn}, Recall: {recall_knn}, F1 Score: {f1_knn}")


# Random Forest
# Select relevant features for Random Forest
features = [
    'artist_popularity',
    'album_popularity',
    'genre_popularity',
    'user_listen_frequency',
    'favorite_artist',
    'recent_trend',
    'artist_loyalty'
]

df_rf = new_df[features + ['is_listened']]  # Using new_df instead of df
X_rf = df_rf.drop(['is_listened'], axis=1)
y_rf = df_rf['is_listened']

# Debugging steps
print("GridSearchCV type:", type(sk_ms.GridSearchCV))
print("RandomForestClassifier type:", type(sk_ensemble.RandomForestClassifier))

X_train_rf, X_test_rf, y_train_rf, y_test_rf = sk_ms.train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

# Debugging (Check Type Information)
print("Type of X_train_rf:", type(X_train_rf))
print("Type of y_train_rf:", type(y_train_rf))
print("Is X_train_rf None?", X_train_rf is None)
print("Is y_train_rf None?", y_train_rf is None)

# Hyperparameter Random Forest
# Random Forest hyperparameter tuning
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

# Instantiate GridSearchCV
grid_rf = sk_ms.GridSearchCV(sk_ensemble.RandomForestClassifier(), param_grid=param_grid_rf, cv=3)

# Fitting the model
grid_rf.fit(X_train_rf, y_train_rf)

# Get the best estimator
clf_rf = grid_rf.best_estimator_

# Fit the best estimator to the training data
clf_rf.fit(X_train_rf, y_train_rf)

# Make predictions on the testing data
predictions_rf = clf_rf.predict(X_test_rf)

# Evaluate Random Forest
# Calculate RMSE
rmse_rf = sk_metrics.mean_squared_error(y_test_rf, predictions_rf, squared=False)

# Calculate Precision and Recall
threshold_rf = 0.5  # You can adjust this threshold as needed
binary_predictions_rf = (predictions_rf >= threshold_rf).astype(int)

precision_rf = sk_metrics.precision_score(y_test_rf, binary_predictions_rf)
recall_rf = sk_metrics.recall_score(y_test_rf, binary_predictions_rf)

# Calculate F1 Score
f1_rf = sk_metrics.f1_score(y_test_rf, binary_predictions_rf)



# Hybrid Model
pred_svd = np.array([pred.est for pred in predictions_svd])
pred_knn = np.array([pred.est for pred in predictions_knn])

# Assuming predictions_rf is already defined
pred_rf = np.array(predictions_rf)

pred_hybrid = (pred_svd + pred_knn + pred_rf[:len(pred_svd)]) / 3

# Convert to binary for evaluation
pred_binary = [1 if x >= 0.5 else 0 for x in pred_hybrid]
true_values = [pred.r_ui for pred in predictions_svd]
true_binary = [1 if x >= 0.5 else 0 for x in true_values]

# Evaluate Hybrid Model

# Calculate RMSE
rmse_hybrid = sk_mean_squared_error(true_values, pred_hybrid, squared=False)

# Calculate Precision and Recall
precision_hybrid = sk_precision_score(true_binary, pred_binary)
recall_hybrid = sk_recall_score(true_binary, pred_binary)

# Calculate F1 Score
f1_hybrid = sk_f1_score(true_binary, pred_binary)

# Print the Results
print(f"Hybrid_model - RMSE: {rmse_hybrid}, Precision: {precision_hybrid}, Recall: {recall_hybrid}, F1 Score: {f1_hybrid}")



# Print the results

print(f"Baseline - RMSE: {rmse_baseline}, Precision: {precision_baseline}, Recall: {recall_baseline}, F1 Score: {f1_baseline}")
print(f"SVD - RMSE: {rmse_svd}, Precision: {precision_svd}, Recall: {recall_svd}, F1 Score: {f1_svd}")
print(f"KNN - RMSE: {rmse_knn}, Precision: {precision_knn}, Recall: {recall_knn}, F1 Score: {f1_knn}")
print(f"Random Forest - RMSE: {rmse_rf}, Precision: {precision_rf}, Recall: {recall_rf}, F1 Score: {f1_rf}")
print(f"Hybrid_model - RMSE: {rmse_hybrid}, Precision: {precision_hybrid}, Recall: {recall_hybrid}, F1 Score: {f1_hybrid}")




# Test the recommender system

# Filter users with at least 50 interactions
#user_interactions = df['user_id'].value_counts()
#filtered_users = user_interactions[user_interactions >= 50].index.tolist()

# Assuming pred_hybrid is your hybrid model's predictions
#def get_top_recommendations_for_user(user_id):
#   user_predictions = pred_hybrid[user_id]
#   top_n_indices = np.argsort(user_predictions)[::-1][:5]  # Get the indices of top 5 recommendations
#   top_n_recommendations = [all_unique_media_ids[i] for i in top_n_indices]
#   return user_id, top_n_recommendations

# Initialize list to store recommendations
#recommendations = {}
#is_first_track_recommended = []

# List of all unique media IDs
#all_unique_media_ids = df['media_id'].unique().tolist()

# Parallelize the process for filtered users
#with ThreadPoolExecutor() as executor:
#      for user_id, top_n_recommendations in executor.map(get_top_recommendations_for_user, filtered_users):
#      recommendations[user_id] = top_n_recommendations
#      true_first_track = omitted_tracks[omitted_tracks['user_id'] == user_id]['media_id'].values[0]
#      is_first_track_recommended.append(1 if true_first_track in top_n_recommendations else 0)

# Calculate the percentage of filtered users for whom the first track was recommended
#percentage_recommended = sum(is_first_track_recommended) / len(is_first_track_recommended) * 100
#print(f"The first track was recommended for {percentage_recommended}% of filtered users with at least 50 interactions.")
