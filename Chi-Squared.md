# Chi-Squared
The chi-squared test is a statistical method used to assess the independence or association between discrete variables, including categorical variables. It is employed when analyzing data in the form of a contingency table, where the observed frequencies are compared to the expected frequencies under the assumption of independence between the variables. The test helps determine whether the observed distribution significantly differs from what would be expected if the variables were independent, making it suitable for detecting associations in categorical data. 

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def chi_squared_plot_stat(df,category_column,outcome_column ):
    sns.countplot(x=outcome_column, hue=category, data=df)

    # Adding labels and title
    plt.xlabel(f'{outcome_column}')
    plt.ylabel('Count')
    plt.title(f'Count of {category} in Each {outcome_column}')

    # Show the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    import pandas as pd
    from scipy.stats import chi2_contingency

    # Assuming your DataFrame is named 'df'
    # Adjust column names accordingly if they are different in your DataFrame

    # Create a contingency table
    contingency_table = pd.crosstab(df[outcome_column], df[category])


    # Assuming df is your DataFrame and outcome_column, category are your column names
    confusion_matrix = pd.crosstab(df[outcome_column], df[category])

    # Plotting the confusion matrix with light colors
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap="icefire", fmt='g', cbar=False)

    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    alpha = 0.05
    if p < alpha:
        print(f"There IS a significant association between {outcome_column} and {category} based on Chi Squared test. P-value: {p}")
    else:
        print(f"There is NO significant association between {outcome_column} and {category}  based on Chi Squared test. P-value: {p}")

        
    print('\n')    
    print( pd.crosstab(df[outcome_column], df[category]))
```

chi_squared_plot_stat(dmm_taxa_map,'birth_mode','dmm_clusters_taxa')
