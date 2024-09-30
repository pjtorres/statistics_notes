# GLM

```python
df_in = mr_clr(relativeabun(merged_sub_keggl3_4))

# Reindex df_in to match the sample_ids in metadata
df_in = df_in.reindex(ambrosia_meta_expanded['sample_id'])

# Create an empty list to store GLM results
glm_results_wald = []

# Loop through each feature (column) in df_in
for col in df_in.columns:
    # Combine df_in (abundance data) with metadata into one DataFrame
    df_combined = pd.merge(df_in.reset_index()[['sample_id', col]], ambrosia_meta_expanded[['sample_id', 'dmm_cluster', 'gender']])
    
    # Clean up column names
    df_combined.columns = df_combined.columns.str.replace('-', '_').str.replace(' ', '_').str.replace('(', '_').str.replace(')', '_').str.replace('/', '_')

    # Replace hyphens and spaces in the specified column name
    col2 = col.replace('-', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_')
    
    # Fit the GLM model: abundance ~ dmm_cluster + gender
    model = smf.glm(formula=f'{col2} ~ C(dmm_cluster) + C(gender)', data=df_combined, family=sm.families.Gaussian())
    result = model.fit()

    # Perform Wald test for the effect of dmm_cluster (testing if all dmm_cluster coefficients are 0)
    wald_test = result.wald_test_terms(['C(dmm_cluster)'])
     # Extract Wald test p-value
    wald_pvalue = wald_test.table['pvalue'].values[0].item()
    # Store p-values and other statistics
    glm_results_wald.append({
        'feature': col,
        'pvalue_dmm_cluster_1_vs_2': result.pvalues['C(dmm_cluster)[T.2]'],  # Cluster 2 vs Cluster 1
        'pvalue_dmm_cluster_1_vs_3': result.pvalues['C(dmm_cluster)[T.3]'],  # Cluster 3 vs Cluster 1
        'pvalue_gender': result.pvalues['C(gender)[T.2]'],  # Assuming gender is binary 'Male'/'Female'
        'wald_pvalue_dmm_cluster':wald_pvalue,  # Overall Wald test p-value for dmm_cluster
        'coef_dmm_cluster_1_vs_2': result.params['C(dmm_cluster)[T.2]'],  
        'coef_dmm_cluster_1_vs_3': result.params['C(dmm_cluster)[T.3]'],  
        'coef_gender': result.params['C(gender)[T.2]']
    })
    
# Convert results into a DataFrame
glm_df_wald = pd.DataFrame(glm_results_wald)

# Multiple testing correction for p-values (optional)
glm_df_wald['pvalue_corr_dmm_cluste_1vs_2'] = multipletests(glm_df_wald['pvalue_dmm_cluster_1_vs_2'],
                                                  alpha=0.05,
                                                  method='fdr_bh')[1]
glm_df_wald['pvalue_corr_dmm_cluste_1vs_3'] = multipletests(glm_df_wald['pvalue_dmm_cluster_1_vs_3'],
                                                  alpha=0.05,
                                                  method='fdr_bh')[1]

glm_df_wald['pvalue_corr_gender'] = multipletests(glm_df_wald['pvalue_gender'],
                                             alpha=0.05,
                                             method='fdr_bh')[1]
glm_df_wald['pvalue_corr_wald_dmm_cluster'] = multipletests(glm_df_wald['wald_pvalue_dmm_cluster'],
                                             alpha=0.05,
                                             method='fdr_bh')[1]

# Display the results
glm_df_wald.head()

```

# ANCOVA
```python
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import f_oneway

metadata = ambrosia_meta_expanded
df_in = mr_clr(relativeabun(merged_sub_keggl3_4))

# Reindex df_in to match the sample_ids in metadata
df_in = df_in.reindex(metadata['sample_id'])

df_ra = relativeabun(merged_sub_keggl3_4)
df_ra = df_ra.reindex(metadata['sample_id'])

namee = []
pva = []
stata = []
a = []
b = []
c = []
ra = []
rb = []
rc = []
se1 = []
se2 = []
se3 = []
ancova_results_wald = []

for col in df_in.columns:
    # Combine df_in (abundance data) with metadata into one DataFrame
    df_combined = pd.merge(df_in.reset_index()[['sample_id', col]], 
                           ambrosia_meta_expanded[['sample_id', 'dmm_cluster', 'gender']], 
                           on='sample_id')

    df_relative = pd.merge(df_ra.reset_index()[['sample_id', col]], 
                           ambrosia_meta_expanded[['sample_id', 'dmm_cluster', 'gender']], 
                           on='sample_id')
    
    # Clean up column names
    df_combined.columns = df_combined.columns.str.replace('-', '_').str.replace(' ', '_').str.replace('(', '_').str.replace(')', '_').str.replace('/', '_')

    # Replace hyphens and spaces in the specified column name
    col2 = col.replace('-', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_')
    
    # Fit the ANCOVA model: abundance ~ dmm_cluster + gender
    model = smf.ols(formula=f'{col2} ~ C(dmm_cluster) + C(gender)', data=df_combined).fit()

    # Perform Wald test for the effect of dmm_cluster (testing if all dmm_cluster coefficients are 0)
    wald_test = model.wald_test_terms(['C(dmm_cluster)'])
    
    # Extract Wald test p-value
    wald_pvalue = wald_test.table['pvalue'].values[0].item()

    # Calculate means, medians, and standard errors for each cluster
    grouped = df_combined.groupby('dmm_cluster')[col2]
    grouped_ra = df_relative.groupby('dmm_cluster')[col]
    
    
    median_a = grouped.median().get(0, np.nan)
    median_b = grouped.median().get(1, np.nan)
    median_c = grouped.median().get(2, np.nan)

    se_a = grouped.std().get(0, np.nan) / np.sqrt(grouped.count().get(0, 1))
    se_b = grouped.std().get(1, np.nan) / np.sqrt(grouped.count().get(1, 1))
    se_c = grouped.std().get(2, np.nan) / np.sqrt(grouped.count().get(2, 1))
    
    mean_a = grouped_ra.mean().get(0, np.nan)
    mean_b = grouped_ra.mean().get(1, np.nan)
    mean_c = grouped_ra.mean().get(2, np.nan)
    
    # Append the calculated values
    a.append(median_a)
    b.append(median_b)
    c.append(median_c)
    
    se1.append(se_a)
    se2.append(se_b)
    se3.append(se_c)
    
    ra.append(mean_a)
    rb.append(mean_b)
    rc.append(mean_c)
    
    namee.append(col)
    
    # Store p-values and other statistics
    ancova_results_wald.append({
        'feature': col,
        'pvalue': wald_pvalue,
        'coef_dmm_cluster_1_vs_2': model.params.get('C(dmm_cluster)[T.2]', np.nan),  
        'coef_dmm_cluster_1_vs_3': model.params.get('C(dmm_cluster)[T.3]', np.nan),  
    })

# Create DataFrame for results
df = pd.DataFrame(ancova_results_wald)

# Correct p-values
df['pvalue_corr'] = multipletests(df['pvalue'], alpha=0.05, method='fdr_bh')[1]

# Add means, medians, and standard errors
df['ra_mean_dmm1'] = ra
df['ra_mean_dmm2'] = rb
df['ra_mean_dmm3'] = rc
df['median_dmm1'] = a
df['median_dmm2'] = b
df['median_dmm3'] = c
df['se_dmm1'] = se1
df['se_dmm2'] = se2
df['se_dmm3'] = se3

# Final DataFrame
anova_df = df

```
