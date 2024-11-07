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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import numpy as np
import re

metadata = ambrosia_meta_expanded
df_in = mr_clr(relativeabun(merged_sub_name3))

# Reindex df_in to match the sample_ids in metadata
df_in = df_in.reindex(metadata['sample_id'])

df_ra = relativeabun(merged_sub_name3)
df_ra = df_ra.reindex(metadata['sample_id'])

namee = []
pva = []
stata = []
medians = {}
means = {}
ses = {}
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
    df_combined.columns = df_combined.columns.str.replace(r'[^a-zA-Z0-9]', '_', regex=True) #df_combined.columns.str.replace('-', '_').str.replace(' ', '_').str.replace('(', '_').str.replace(')', '_').str.replace('/', '_').str.replace(',', '_').str.replace('--', '_')
    df_combined.columns = df_combined.columns.str.replace(r'^(\d)', r'col_\1', regex=True)
    # Replace hyphens and spaces in the specified column name
    col2 =  df_combined.columns[1]#re.sub(r'[^a-zA-Z0-9]', '_', col) #.replace('-', '_').replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_').replace(',', '_').replace('--', '_')

    # Fit the ANCOVA model: abundance ~ dmm_cluster + gender
    model = smf.ols(formula=f'{col2} ~ C(dmm_cluster) + C(gender)', data=df_combined).fit()

    # Perform Wald test for the effect of dmm_cluster (testing if all dmm_cluster coefficients are 0)
    wald_test = model.wald_test_terms(['C(dmm_cluster)'])
    
    # Extract Wald test p-value
    wald_pvalue = wald_test.table['pvalue'].values[0].item()

    # Group data by 'dmm_cluster'
    grouped = df_combined.groupby('dmm_cluster')[col2]
    grouped_ra = df_relative.groupby('dmm_cluster')[col]

    # Initialize dictionaries to store results for the current feature
    medians[col] = {}
    means[col] = {}
    ses[col] = {}

    # Loop over unique dmm_cluster values dynamically
    for cluster in df_combined['dmm_cluster'].unique():
        # Calculate median, standard error, and mean for each cluster
        medians[col][cluster] = grouped.median().get(cluster, np.nan)
        ses[col][cluster] = grouped.std().get(cluster, np.nan) / np.sqrt(grouped.count().get(cluster, 1))
        means[col][cluster] = grouped_ra.mean().get(cluster, np.nan)
    
    namee.append(col)

    # Store p-values and coefficients in the ancova_results_wald list
    ancova_results_wald.append({
        'feature': col,
        'pvalue': wald_pvalue,
        'coef_dmm_cluster_1_vs_2': model.params.get('C(dmm_cluster)[T.2]', np.nan),  
        'coef_dmm_cluster_1_vs_3': model.params.get('C(dmm_cluster)[T.3]', np.nan),  
    })

# Create DataFrame for results
df = pd.DataFrame(ancova_results_wald)

# Correct p-values using FDR
df['pvalue_corr'] = multipletests(df['pvalue'], alpha=0.05, method='fdr_bh')[1]

# Add means, medians, and standard errors for all clusters dynamically
for cluster in df_combined['dmm_cluster'].unique():
    df[f'ra_mean_dmm{cluster}'] = df['feature'].map(lambda col: means[col].get(cluster, np.nan))
    df[f'median_dmm{cluster}'] = df['feature'].map(lambda col: medians[col].get(cluster, np.nan))
    df[f'se_dmm{cluster}'] = df['feature'].map(lambda col: ses[col].get(cluster, np.nan))

# Final DataFrame
ancova_df_sub_name3 = df

```

```python
#figure
df = ancova_df_sub_name3[ancova_df_sub_name3['pvalue_corr']<=0.00001]

# main column you would like to organize to
main_col = 'median_dmm1'
# getting list but skipping first few which are stats related
all_col = list(df.columns)[3:]


passed_stats = df
# passed_stats = df[df['pvalue_corr'] <= 0.01]
def column_to_list(df, colint):
    newlist=[]

    for i in df[colint]:
        num = (df[i].values[0])
        newlist.append(num)
    return (newlist)


features =[]

# CHANGE THIS IF YOU WAN TO REVERSE THE ASCENDING AS IS THE CASE FOR CLR VALUES
df2=passed_stats.sort_values(main_col, ascending=True).copy()
######

df2 = df2#.reset_index()
col = df2[all_col]
colval =  df2[all_col].values.tolist()

feature = df2['feature']
features = feature.tolist()


import math
f = plt.figure(figsize=(20,20))
count=0
for i in range(0,len(features)):
    f  = features[i]
    f_df = df2[df2['feature']==f]
    labels = ['median_dmm1','median_dmm2','median_dmm3']
    colval =  column_to_list(f_df,['median_dmm1','median_dmm2','median_dmm3'])
    colva_se =  column_to_list(f_df,['se_dmm1', 'se_dmm2','se_dmm3'])
    colors = np.array(["grey","gold","orange"])#,"blue"])
    for ci in range(0,len(colval)):
        plt.scatter(x = colval[ci], y=f, c= colors[ci], label = labels[ci] if count ==0 else "", facecolors='none',
            linewidths=2, s =300 )
        plt.hlines(y=f, xmin=colval[ci]-colva_se[ci], xmax=colval[ci]+colva_se[ci], colors=colors[ci], linewidth=2)

    count+=1
colors=["grey","gold","orange"]


# Set larger font size for labels and title
plt.xlabel('clr_val', fontsize=14)
plt.title("Cazy sub_name3  ; pvalcorr <= 0.00001 ", fontsize=16)
plt.legend(bbox_to_anchor=(0.5, 1.1), loc="center", fontsize=16)

# plt.savefig(outdir+'full_cazy_substrate_high_level.svg', format='svg')

plt.show()
```

# ANCOVA 2 daling with really weird characters in data that we encounter iin kegg names quite a bit
Currently, I'm working with DMM clusters, but this needs to be updated to automatically handle any number of conditional variables I want.

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import numpy as np
import re

metadata = mapping_filev3
df_in = species_clr
df_ra = species_ra

# Reindex df_in to match the sample_ids in metadata
df_in = df_in.reindex(mapping_filev3['sample_id'])

df_ra = df_ra.reindex(mapping_filev3['sample_id'])

namee = []
pva = []
stata = []
medians = {}
means = {}
ses = {}
ancova_results_wald = []



for col in df_in.columns:
    # Combine df_in (abundance data) with metadata into one DataFrame
    df_combined = pd.merge(df_in.reset_index()[['sample_id', col]], 
                           metadata[['sample_id', 'dmm_cluster', 'gender']], 
                           on='sample_id')

    df_relative = pd.merge(df_ra.reset_index()[['sample_id', col]], 
                           metadata[['sample_id', 'dmm_cluster', 'gender']], 
                           on='sample_id')
    print(col)
    # Clean column names in the DataFrame, replacing spaces, periods, '>' characters, '+' characters, commas, semicolons, and other special characters with underscores
    df_combined.columns = df_combined.columns.str.replace(r'[\s,;]', '_', regex=True).str.replace(r'[-()/\.>,+]', '_', regex=True).str.replace(r'[\[\]:"]', '_', regex=True).str.replace(r'^(?=\d)', '_', regex=True).str.replace(r"'", '_', regex=True)
    
    # Clean a specific column name
    col2 = re.sub(r'[\s,;]', '_', col)  # Replace spaces, commas, and semicolons first
    col2 = re.sub(r'[-()/\.>,+]', '_', col2)  # Replace other special characters
    col2 = re.sub(r'[\[\]:"]', '_', col2)  # Replace any remaining unwanted characters
    col2 = re.sub(r'^(?=\d)', '_', col2)
    col2 = re.sub(r"'", '_', col2)
    # print(col2)
    # print(df_combined[col2])
    # Fit the ANCOVA model: abundance ~ dmm_cluster + gender
    model = smf.ols(formula=f'{col2} ~ C(dmm_cluster) + C(gender)', data=df_combined).fit()

    # Perform Wald test for the effect of dmm_cluster (testing if all dmm_cluster coefficients are 0)
    wald_test = model.wald_test_terms(['C(dmm_cluster)'])
    
    # Extract Wald test p-value
    wald_pvalue = wald_test.table['pvalue'].values[0].item()

    # Group data by 'dmm_cluster'
    grouped = df_combined.groupby('dmm_cluster')[col2]
    grouped_ra = df_relative.groupby('dmm_cluster')[col]

    # Initialize dictionaries to store results for the current feature
    medians[col] = {}
    means[col] = {}
    ses[col] = {}

    # Loop over unique dmm_cluster values dynamically
    for cluster in df_combined['dmm_cluster'].unique():
        # Calculate median, standard error, and mean for each cluster
        medians[col][cluster] = grouped.median().get(cluster, np.nan)
        ses[col][cluster] = grouped.std().get(cluster, np.nan) / np.sqrt(grouped.count().get(cluster, 1))
        means[col][cluster] = grouped_ra.mean().get(cluster, np.nan)
    
    namee.append(col)

    # Store p-values and coefficients in the ancova_results_wald list
    ancova_results_wald.append({
        'feature': col,
        'pvalue': wald_pvalue,
        'coef_dmm_cluster_1_vs_2': model.params.get('C(dmm_cluster)[T.2]', np.nan),  
        'coef_dmm_cluster_1_vs_3': model.params.get('C(dmm_cluster)[T.3]', np.nan),  
    })

# Create DataFrame for results
df = pd.DataFrame(ancova_results_wald)

# Correct p-values using FDR
df['pvalue_corr'] = multipletests(df['pvalue'], alpha=0.05, method='fdr_bh')[1]

# Add means, medians, and standard errors for all clusters dynamically
for cluster in df_combined['dmm_cluster'].unique():
    df[f'ra_mean_dmm{cluster}'] = df['feature'].map(lambda col: means[col].get(cluster, np.nan))
    df[f'median_dmm{cluster}'] = df['feature'].map(lambda col: medians[col].get(cluster, np.nan))
    df[f'se_dmm{cluster}'] = df['feature'].map(lambda col: ses[col].get(cluster, np.nan))

# Add 'higher_in' column based on the maximum ra_mean for each feature
df['higher_in'] = df.apply(
    lambda row: f'dmm{np.argmax([row["ra_mean_dmm1"], row["ra_mean_dmm2"], row["ra_mean_dmm3"]]) + 1}', 
    axis=1
)


# Final DataFrame
ancova_df = df
```
