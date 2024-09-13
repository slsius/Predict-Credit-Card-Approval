# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None) 

# Replace space with NaN
cc_apps.replace('', np.nan, inplace=True)
cc_apps.replace('?', np.nan, inplace=True)

# Fill NaN value with Most Frequent if type is object or Mean value if numeric type
for column in cc_apps.columns:
    if cc_apps[column].dtype == 'object':  # If column is of object type
        most_frequent_value = cc_apps[column].mode()[0]
        cc_apps[column] = cc_apps[column].fillna(most_frequent_value)
    else:   # If column is other type
        mean_value = cc_apps[column].mean()
        cc_apps[column] = cc_apps[column].fillna(mean_value)

# Expand categorical value to 1 / 0
cc_dummies = pd.get_dummies(cc_apps, drop_first=True)

# Define Target data
x = cc_dummies.iloc[:, :-1].values
y = cc_dummies.iloc[:, -1].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define pipelines for each model
pipe_logreg = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression())])
pipe_lasso = Pipeline([('scaler', StandardScaler()), ('lasso', LogisticRegression(penalty='l1', solver='saga', max_iter=10000))])
pipe_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', LogisticRegression(penalty='l2', solver='saga', max_iter=10000))])

# Define parameter grids for RandomizedSearchCV
param_dist_logreg = {
    'logreg__C': uniform(loc=0, scale=4)  # Logistic Regression (no penalty, regularization strength)
}

param_dist_lasso = {
    'lasso__C': uniform(loc=0, scale=4)  # Lasso (L1 penalty, regularization strength)
}

param_dist_ridge = {
    'ridge__C': uniform(loc=0, scale=4)  # Ridge (L2 penalty, regularization strength)
}

# Set up RandomizedSearchCV for each model
random_search_logreg = RandomizedSearchCV(pipe_logreg, param_distributions=param_dist_logreg, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_lasso = RandomizedSearchCV(pipe_lasso, param_distributions=param_dist_lasso, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_ridge = RandomizedSearchCV(pipe_ridge, param_distributions=param_dist_ridge, n_iter=20, cv=5, scoring='accuracy', random_state=42)

# Fit the models
random_search_logreg.fit(X_train, y_train)
random_search_lasso.fit(X_train, y_train)
random_search_ridge.fit(X_train, y_train)

# Get the best scores and parameters for each model
logreg_best_params = random_search_logreg.best_params_
logreg_best_score = random_search_logreg.best_score_
lasso_best_params = random_search_lasso.best_params_
lasso_best_score = random_search_lasso.best_score_
ridge_best_params = random_search_ridge.best_params_
ridge_best_score = random_search_ridge.best_score_

print(f"Logistic Regression Best Params: {logreg_best_params}, Best CV Score: {logreg_best_score}")
print(f"Lasso Regression Best Params: {lasso_best_params}, Best CV Score: {lasso_best_score}")
print(f"Ridge Regression Best Params: {ridge_best_params}, Best CV Score: {ridge_best_score}")

# Compare test scores on the test set for each model
logreg_test_score = random_search_logreg.score(X_test, y_test)
lasso_test_score = random_search_lasso.score(X_test, y_test)
ridge_test_score = random_search_ridge.score(X_test, y_test)

print(f"Test Accuracy (Logistic Regression): {logreg_test_score}")
print(f"Test Accuracy (Lasso Regression): {lasso_test_score}")
print(f"Test Accuracy (Ridge Regression): {ridge_test_score}")

# Confusion matrices for each model
logreg_conf_matrix = confusion_matrix(y_test, random_search_logreg.predict(X_test))
lasso_conf_matrix = confusion_matrix(y_test, random_search_lasso.predict(X_test))
ridge_conf_matrix = confusion_matrix(y_test, random_search_ridge.predict(X_test))

print("\nConfusion Matrix (Logistic Regression):\n", logreg_conf_matrix)
print("Confusion Matrix (Lasso Regression):\n", lasso_conf_matrix)
print("Confusion Matrix (Ridge Regression):\n", ridge_conf_matrix)
