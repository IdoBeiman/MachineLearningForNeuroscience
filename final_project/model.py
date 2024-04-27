import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
from feature_selection import find_meaningful_features
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator

from pre_process import pre_process_dataset


def perform_grid_search(model: BaseEstimator, X, y, param_grid, num_folds=5):
    # Setup cross-validation
    kf  = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    scoring = {'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False),
               'R2': 'r2'}

    # Initialize the GridSearchCV object with multiple scoring
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring=scoring, refit='RMSE', verbose=1, return_train_score=True )   
    
    # Perform the grid search
    grid_search.fit(X, y)
    
    # Output the best parameters and the corresponding scores
    print("Best parameters:", grid_search.best_params_)
    best_rmse = -grid_search.cv_results_['mean_test_RMSE'][grid_search.best_index_]
    best_r2 = grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]
    print("Best cross-validation RMSE:", best_rmse)
    print("Best cross-validation R²:", best_r2)
    
    # Return the best model and its score
    return grid_search.best_estimator_, best_rmse, best_r2




def train_and_evaluate_model_with_cv(data):
    y = data['citation_count']
    X = data.drop('citation_count', axis=1)
    X.head(5).to_csv("data/X.csv")
    model = Ridge(random_state=42)
    param_grid = {
        'alpha': [1,10,100,1000],
    }    
    # Perform grid search to find the best model and RMSE
    best_model, best_rmse, best_r2 = perform_grid_search(model, X, y, param_grid)
    
    # Output the results
    print("Best Model Parameters:", best_model.get_params())





# filtered_df = pre_process_dataset("data/data-neurosynth_version-7_features_with_citations.csv")
filtered_df = pd.read_csv("data/filtered_df.csv")
data = find_meaningful_features(filtered_df, num_features=200)
train_and_evaluate_model_with_cv(data)
