
Logistic Regression is a statistical model commonly used for binary classification (though it can also be applied when there are more than two class variables), predicting the probability of an instance belonging to a specific class. Its linearity assumption models the log-odds linearly, providing interpretable coefficients. Logistic Regression assumes independence of errors, no multicollinearity, and homoscedasticity. Compared to other models, it is linear and yields probabilistic outputs, making it suitable for scenarios like spam or fraud detection. It excels in interpretability, and when combined with L1 regularization (LASSO), it becomes adept at automatic feature selection by shrinking some coefficients to zero, offering a sparse solution that mitigates overfitting. While logistic regression is effective, its applicability hinges on meeting its assumptions and considering the characteristics of the data at hand.

In this example, we would begin with a pandas dataframe where each column represents a gene function, and each row corresponds to a sample, as illustrated below.

1. Dataframe example (gc_clusters_meta_pivot_filtered_rav2_cluster)
   
| sample_name    | feature1 | feature2 | feature1 | Class | 
| -------- | ------- | ------- | ------- | ------- |
|PB1  | 0.43    |  1.8 | 0.4    |  C1 |
| PB2 | -1.2    | 0.8 | 0.3    |  C1 |
| PB3    | 1.5    | 1 | -0.43    |  C2 |

2. Create a stratified test and training dataset.

```python
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in split.split(gc_clusters_meta_pivot_filtered_rav2_cluster,gc_clusters_meta_pivot_filtered_rav2_cluster['Class']):
    strat_train_set = gc_clusters_meta_pivot_filtered_rav2_cluster.loc[train_index].set_index('sample_name')
    strat_test_set = gc_clusters_meta_pivot_filtered_rav2_cluster.loc[test_index].set_index('sample_name')
    
    
group_feat = 'Class'
group_labels=strat_train_set.reset_index()[['sample_name',group_feat]]
group_labels = pd.Series(group_labels[group_feat].values, index=group_labels['sample_name'])

gc_ra = strat_train_set.drop(columns=[group_feat]).reset_index().rename(columns={"sample_name": ""}).set_index('')
X=gc_ra
y=group_labels
```

3. Optional (Use GridSearchCV to find the optimal parameters for your model):
   
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lgr', LogisticRegression( penalty='l1'))
])


# Define the parameter grid for grid search
param_grid = {

    'lgr__C': [ 0.01, 0.1, 1, 10],
    'lgr__class_weight': [None, 'balanced'],
    'lgr__solver' : ['liblinear'],
    'lgr__multi_class' : ['auto']

}

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=7)

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X, y)

# Get the best pipeline from grid search
pipeline = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)
```

4. Run your logistic regression model and track key metrics, such as the mean accuracy score and mean coefficients for each significant variable. Record important variables along with their coefficients (excluding those with a coefficient of 0). Also, keep a count of the frequency of each variable to facilitate the assessment of feature stability later on.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import numpy as np
import pandas as pd

# Define the pipeline (with optional scaler)
pipeline = Pipeline([
    # You might want to add StandardScaler if your features vary in scale.
    # ('scaler', StandardScaler()),
    ('lgr', LogisticRegression(C=2, max_iter=5000, penalty='l1', solver='liblinear',class_weight='balanced'))
])

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=20)

# Initialize containers for results
accuracies = []
confusion_matrices = []
feature_importances = []
all_selected_features = []

# Perform cross-validation
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Store accuracy and confusion matrix
    accuracies.append(accuracy_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
    # Get the feature importances (non-zero coefficients)
    coef = pipeline.named_steps['lgr'].coef_
    
    # If multiclass, average coefficients across classes
    avg_coef = np.mean(coef, axis=0)
    
    selected_features = X_train.columns[avg_coef != 0]
    feature_scores = avg_coef[avg_coef != 0]
    
    all_selected_features.append(selected_features)
    
    # Store the feature importances
    feature_importances.append((selected_features, feature_scores))

# Calculate the mean accuracy
mean_accuracy = np.mean(accuracies)
print("Mean Accuracy:", mean_accuracy)

# Consolidate feature importances across folds
most_important_features = {}
for importance in feature_importances:
    for i in range(len(importance[0])):
        feature = importance[0][i]
        score = importance[1][i]
        if feature in most_important_features:
            most_important_features[feature].append(score)
        else:
            most_important_features[feature] = [score]

# Calculate the average score for each feature
for feature in most_important_features:
    most_important_features[feature] = np.mean(most_important_features[feature])

# Sort the features by score
sorted_features = sorted(most_important_features.items(), key=lambda x: x[1], reverse=True)

# Print the most important features and their average coefficients
print("Most Important Features with Average Coefficients:")
for feature, avg_score in sorted_features:
    print(f"{feature} - {avg_score}")

# Count feature frequency across folds
feature_frequencies = Counter(feature for sublist in all_selected_features for feature in sublist)

# Sort features by frequency
sorted_features_frequency = sorted(feature_frequencies.items(), key=lambda x: x[1], reverse=True)

# Convert to DataFrame for better visualization
freq_df = pd.DataFrame(sorted_features_frequency, columns=['feature', 'Frequency'])
coeff_df = pd.DataFrame(sorted_features, columns=['feature', 'Average Coeff'])

# Merge frequency and coefficient DataFrames
sorted_frequency_feat = pd.merge(coeff_df, freq_df, on='feature')

# Print the final merged DataFrame
print(sorted_frequency_feat)

# Calculate and print the mean confusion matrix
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
print("Mean Confusion Matrix:")
print(mean_confusion_matrix)

```
```python
# save model
import pickle


# Save the model to a file using pickle
with open('lgr_model.pkl', 'wb') as file:
     pickle.dump(pipeline, file)

```

5. Explore Test Confusion Matrix:

```python
# Calculate the sum of each row
row_sums = mean_confusion_matrix.sum(axis=1)

# Divide each element by the sum of the elements in that row and multiply by 100. This
# is only if you want a percentage. You can use the raw output as well just have to modify a bit
mean_confusion_matrix_percent = (mean_confusion_matrix / row_sums[:, np.newaxis]) * 100

# Plot the mean confusion matrix as percentages
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mean_confusion_matrix_percent, interpolation='nearest', cmap=plt.cm.Reds)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(mean_confusion_matrix_percent.shape[1]),
       yticks=np.arange(mean_confusion_matrix_percent.shape[0]),
       xticklabels=list(y.unique()), yticklabels=list(y.unique()), # note that you will change this for your labels
       title="Training Mean Confusion Matrix (Percentages)",
       ylabel="True label",
       xlabel="Predicted label")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(mean_confusion_matrix_percent.shape[0]):
    for j in range(mean_confusion_matrix_percent.shape[1]):
        ax.text(j, i, format(mean_confusion_matrix_percent[i, j], '.2f') + '%',
                ha="center", va="center",
                color="white" if mean_confusion_matrix_percent[i, j] > mean_confusion_matrix_percent.max() / 2. else "black")
fig.tight_layout()
plt.show()
```

6. Run model on the test dataset you left out

```python
group_feat = 'Class'
group_labels_test=strat_test_set.reset_index()[['sample_name',group_feat]]
group_labels_test = pd.Series(group_labels_test[group_feat].values, index=group_labels_test['sample_name'])

group_labels_test


ko_clr_test = strat_test_set.drop(columns=[group_feat]).reset_index().rename(columns={"sample_name": ""}).set_index('')
X_test_final=ko_clr_test
y_test_final=group_labels_test

probas_ = pipeline.predict_proba(X_test_final)
y_pred_final = pipeline.predict(X_test_final)
y_pred_final
pipeline['lgr'].coef_ 
X_test_final.columns

coeffu = pd.DataFrame(pipeline['lgr'].coef_ , columns=X_test_final.columns).T
coeffu['coeff'] = (coeffu[0] + coeffu[1] +coeffu[2])/3

y_pred_final = pipeline.predict(X_test_final)
accuracy = accuracy_score(y_test_final, y_pred_final)
print(accuracy)
# Plot the mean confusion matrix
confusion_matrix_data_final = confusion_matrix(y_test_final, y_pred_final)

confusion_matrix_data_final
row_sums = confusion_matrix_data_final.sum(axis=1)

# Divide each element by the sum of the elements in that row and multiply by 100
confusion_matrix_percent = (confusion_matrix_data_final / row_sums[:, np.newaxis]) * 100
confusion_matrix_data_final = confusion_matrix_percent
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion_matrix_data_final, interpolation='nearest', cmap=plt.cm.Reds)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(mean_confusion_matrix.shape[1]),
       yticks=np.arange(mean_confusion_matrix.shape[0]),
       xticklabels=list(y.unique()), yticklabels=list(y.unique()), # this is modified depending on your classes similar to above
       title="Test Mean Confusion Matrix",
       ylabel="True label",
       xlabel="Predicted label")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(confusion_matrix_data_final.shape[0]):
    for j in range(confusion_matrix_data_final.shape[1]):
        ax.text(j, i, format(confusion_matrix_data_final[i, j], '.2f'),
                ha="center", va="center",
                color="white" if confusion_matrix_data_final[i, j] > confusion_matrix_data_final.max() / 2. else "black")
fig.tight_layout()
plt.show()
```

