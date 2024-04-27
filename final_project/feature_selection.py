import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

COLUMN_TO_DROP_FOR_FEATURE_SELECTION= ['Unnamed: 0', 'citation_count','Impact Factor']

def cluster_terms(file_csv):
    # Load and prepare data
    data = pd.read_csv(file_csv)
    tfidf_data = data.drop(['Unnamed: 0', 'citation_count', 'journal'], axis=1)  # assuming these are non-TF-IDF columns

    # Apply NMF for dimensionality reduction
    nmf = NMF(n_components=50, random_state=42)  # choose components based on domain knowledge or experimentation
    W = nmf.fit_transform(tfidf_data)
    H = nmf.components_

    # Normalize the feature matrix W
    W_normalized = normalize(W, norm='l1')

    # Cluster using k-means
    kmeans = KMeans(n_clusters=10, random_state=42)  # adjust clusters as needed
    clusters = kmeans.fit_predict(W_normalized)

    # Attach cluster labels to the original data
    data['Cluster'] = clusters

    # Analyze and label clusters
    for i, cluster in enumerate(kmeans.cluster_centers_):
        top_terms_idx = cluster.argsort()[-10:]  # get indices of top 10 terms for each cluster
        top_terms = tfidf_data.columns[top_terms_idx]
        print(f"Cluster {i} top terms: {top_terms}")

    # Save or return the updated DataFrame
    data.to_csv('path_to_clustered_data.csv')  # or return data for further use


def find_meaningful_features(data: pd.DataFrame, num_features=10, ENCODING_PREFIX='enc_'):
    # Check if necessary columns are present
    if 'citation_count' not in data.columns:
        raise ValueError("Data must include 'citation_count' as a column.")
    
    y = data['citation_count']
    
    # Ensure that the column to drop is in the DataFrame
    if not set(COLUMN_TO_DROP_FOR_FEATURE_SELECTION).issubset(data.columns):
        raise ValueError(f"Columns {COLUMN_TO_DROP_FOR_FEATURE_SELECTION} are not all in the DataFrame.")

    # Remove specified columns and one-hot encoded columns for feature selection
    X = data.drop(COLUMN_TO_DROP_FOR_FEATURE_SELECTION, axis=1)
    X = X.loc[:, ~X.columns.str.contains(ENCODING_PREFIX) & ~X.columns.str.contains('Unnamed')]

    # Handle the case where num_features is greater than the number of available features
    max_features = min(num_features, X.shape[1])

    # Initialize and apply SelectKBest
    fs = SelectKBest(score_func=f_regression, k=max_features)
    try:
        X_selected = fs.fit_transform(X, y)
    except ValueError as e:
        raise ValueError(f"Error in feature selection: {e}")

    # Get selected feature names and their scores
    feature_scores = zip(X.columns[fs.get_support()], fs.scores_[fs.get_support()])
    feature_scores = sorted(feature_scores, key=lambda x: x[1], reverse=True)

    print(f"Top {max_features} features selected by f_regression:")
    for feature, score in feature_scores:
        print(f"{feature}: {score:.4f}")

    # Prepare the selected features DataFrame
    selected_features_df = pd.DataFrame(X_selected, columns=X.columns[fs.get_support()])

    # Rejoin selected columns that were dropped
    selected_columns = data[COLUMN_TO_DROP_FOR_FEATURE_SELECTION].join(data.loc[:, data.columns.str.contains(ENCODING_PREFIX)])
    selected_features_df = selected_features_df.join(selected_columns.reset_index(drop=True))
    selected_features_df.head(5).to_csv("data/filtered_df_5_selected.csv")
    return selected_features_df


def apply_lasso_feature_selection(data, target_column, test_size=0.2, random_state=42):
    # Split the data into features and the target variable
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Initialize LassoCV: 10-fold cross-validation
    lasso = LassoCV(cv=10, random_state=random_state, max_iter=10000)
    # Fit LassoCV to the training data
    lasso.fit(X_train, y_train)
    # Determine which features were selected by Lasso (non-zero coefficients)
    feature_mask = lasso.coef_ != 0
    selected_features = X_train.columns[feature_mask]

    # Create a new DataFrame with only selected features plus the target column
    filtered_data = data.loc[:, selected_features.tolist() + [target_column]]

    # Predict on the testing set using only the selected features
    y_pred = lasso.predict(X_test.loc[:, selected_features])

    # Calculate RMSE and R-squared
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("Optimal alpha value:", lasso.alpha_)
    print("Test RMSE:", rmse)
    print("Test R²:", r2)
    print("Selected features:", selected_features)

    # Return the filtered DataFrame with selected features and performance metrics
    filtered_data.to_csv("data/selected_features.csv") 

    return filtered_data
