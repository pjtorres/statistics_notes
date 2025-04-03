import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import re  # For column name cleaning

def clean_column_names(df):
    """Replaces spaces and special characters in column names with underscores for GLM compatibility."""
    clean_mapping = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns}
    return df.rename(columns=clean_mapping), clean_mapping

def plot_boxplot_with_tests(data, var1, var2, covariates=None, test_type='mannwhitneyu', palette=None, figsize=(12, 8)):
    """
    Enhanced function that checks for column existence before proceeding,
    ensures correct formatting, and handles ANCOVA, GLM, and other tests.
    """

    # Standardize and clean column names for safety
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces
    data_cleaned, clean_mapping = clean_column_names(data)

    # Map cleaned names for GLM/ANCOVA models
    if test_type in ['glm', 'np_glm', 'ancova']:
        var1 = clean_mapping.get(var1, var1)
        var2 = clean_mapping.get(var2, var2)
        covariates = [clean_mapping.get(cov, cov) for cov in covariates] if covariates else []
    else:
        data_cleaned = data.copy()

    # Check if all required columns exist
    required_columns = [var1, var2] + (covariates if covariates else [])
    missing_columns = [col for col in required_columns if col not in data_cleaned.columns]
    
    if missing_columns:
        raise KeyError(f"❌ Missing column(s) in data: {missing_columns}")

    # Ensure categorical variables are set to 'category' dtype for GLM/ANCOVA
    for col in [var1] + (covariates if covariates else []):
        if data_cleaned[col].dtype == 'string' or data_cleaned[col].dtype == 'object':
            data_cleaned[col] = data_cleaned[col].astype('category')

    # Ensure numeric values are numeric
    data_cleaned[var2] = pd.to_numeric(data_cleaned[var2], errors='coerce')

    # Drop missing values
    data_cleaned = data_cleaned.dropna(subset=[var1, var2] + (covariates if covariates else []))

    # Error handling for empty data
    if data_cleaned.empty:
        raise ValueError("Error: No valid data remaining after cleaning. Please check your data.")

    # Filter the relevant data
    _scaled = data_cleaned[[var1, var2] + (covariates if covariates else [])].copy()

    # Define category order
    category_order = sorted(_scaled[var1].unique())

    # Set plot style
    sns.set(style='whitegrid')

    # Create the figure
    plt.figure(figsize=figsize)
    ax = sns.boxplot(x=var1, y=var2, data=_scaled, hue=var1,
                     palette=palette if palette else ['#4daf4a', '#377eb8', '#e41a1c'],
                     order=category_order, legend=False)
    
    sns.stripplot(x=var1, y=var2, data=_scaled, color='black',
                  size=5, alpha=0.5, order=category_order)

    # Perform selected test
    pairs = list(combinations(category_order, 2))
    annotations = []
    p_values = []

    # Function to return the significance level as stars
    def get_star_annotation(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    # Perform selected test
    for (cat1, cat2) in pairs:
        group1 = _scaled[_scaled[var1] == cat1][var2].dropna()
        group2 = _scaled[_scaled[var1] == cat2][var2].dropna()

        if len(group1) > 0 and len(group2) > 0:
            if test_type == 'mannwhitneyu':
                _, p_value = mannwhitneyu(group1, group2)
            elif test_type == 'ttest':
                _, p_value = ttest_ind(group1, group2, equal_var=False)
            elif test_type == 'glm':
                formula = f"{var2} ~ {var1} " + (f" + {' + '.join(covariates)}" if covariates else "")
                model = smf.ols(formula, data=_scaled).fit()
                p_value = model.pvalues[var1]
            elif test_type == 'ancova':
                formula = f"{var2} ~ {var1} " + (f" + {' + '.join(covariates)}" if covariates else "")
                model = smf.ols(formula, data=_scaled).fit()
                ancova_results = anova_lm(model, typ=2)
                print(ancova_results)
                p_value = ancova_results["PR(>F)"][var1]
            elif test_type == 'np_glm':
                formula = f"{var2} ~ {var1} " + (f" + {' + '.join(covariates)}" if covariates else "")
                model = smf.rlm(formula, data=_scaled).fit()
                p_value = model.pvalues[var1]
            else:
                raise ValueError("Invalid test_type. Choose 'mannwhitneyu', 'ttest', 'glm', 'np_glm', or 'ancova'.")

            annotations.append((cat1, cat2, p_value))
            p_values.append(p_value)
        else:
            print(f"Skipping {test_type} test for {cat1} vs {cat2} due to insufficient data.")

    # Multiple comparison correction if more than 3 categories
    if len(category_order) >= 3:
        _, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        for idx, (cat1, cat2, _) in enumerate(annotations):
            annotations[idx] = (cat1, cat2, adjusted_p_values[idx])
        print("✅ Multiple comparison correction applied (FDR-BH)")

    # Add annotations
    y_max = _scaled[var2].max()
    y_range = _scaled[var2].max() - _scaled[var2].min()

    for i, (cat1, cat2, p_value) in enumerate(annotations):
        x1, x2 = category_order.index(cat1), category_order.index(cat2)
        y, h, col = y_max + (i + 1) * 0.05 * y_range, 0.02 * y_range, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        ax.text((x1 + x2) * 0.5, y + h, get_star_annotation(p_value), ha='center', color=col, fontsize=25)

    # Set labels
    plt.xlabel(var1, fontsize=28)
    plt.ylabel(var2, fontsize=28)

    # Adjust tick sizes
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)

    # Show plot
    plt.tight_layout()
    plt.show()

    return ax

# Example Usage
#plot_boxplot_with_tests(result_df, 'dmm_cluster', 'Diet Score', covariates=['gender', 'ethnicity', 'sample_storage'], test_type='ancova')
