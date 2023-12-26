# Running Linear Mixed Models

Medium article expanding on the strengths on LMM: https://medium.com/@PedroJTorres_/statistical-insights-addressing-challenges-in-repeated-measures-data-with-linear-mixed-models-41fe4575331c

Linear Mixed Models (LMMs) represent an advanced and versatile tool in modern statistical analysis, particularly adept at handling repeated measure data. Their adaptability in managing unstructured data, incorporation of random effects, and precise modeling of repeated measures render them an essential choice for researchers across various fields seeking accurate analysis of complex and correlated data structures. The significance of LMMs lies in their ability to provide more accurate and reliable inferences in scenarios where traditional methods, such as ANOVA, fall short in handling the complexities of repeated measures.

```python
#This is internaly. For some reason model does not work well if it has weird characters in column names or names start with digits.

metabolomic_df_clrv2 = metabolomic_df_clr.copy()
metabolomic_df_clrv2.columns = metabolomic_df_clrv2.columns.str.replace(' ', '')
metabolomic_df_clrv2.columns = metabolomic_df_clrv2.columns.str.replace('-', '')
metabolomic_df_clrv2.columns = metabolomic_df_clrv2.columns.str.replace('(', '')
metabolomic_df_clrv2.columns = metabolomic_df_clrv2.columns.str.replace(')', '')

def process_column_name(col_name):
    if col_name[0].isdigit(): 
        return 'd' + col_name
    else:
        return col_name

# Apply the custom function to each column name
metabolomic_df_clrv2.columns = metabolomic_df_clrv2.columns.map(process_column_name)

metabolomic_df_clrv2.columns
```


1. Dataframe example (metabolomic_df_clrv2)
   
| metabolite_sub_id    | dmetabolite1 | metabolite2 | 
| -------- | ------- | ------- |
|PB1  | 0.43    |  1.8 |
| PB2 | -1.2    | 0.8 |
| PB3    | 1.5    | 1 |

3. Metadata example (hmo_metadata)
   
| metabolite_sub_id |	inoculation	|sample_name	|media |
| -------- | ------- | -------- | ------- |
| PB1	| Combo_9	| PBT-03139	| HMO|
|PB2	|Combo_9	|PBT-03151	|HMO|
|PB3	|Combo_9	|PBT-03330	|HMO|


3. Do LMM on each metabolite
```python 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

df_in = metabolomic_df_clrv2
hmo_data2 = hmo_metadata.set_index('sample_name')

# Define sample names and probiotic combinations
sample_names = ['PBT-03017', 'PBT-03107', 'PBT-03205', 'PBT-03206', 'PBT-03260', 'PBT-03330']
probiotic_combinations = list(hmo_metadata['inoculation'].unique())

results_list = []  # Create a list to store DataFrames

for metabolite in df_in.columns:
    data = hmo_metadata[(hmo_metadata['sample_name'].isin(sample_names))]
    data = pd.merge(data, df_in.reset_index()[['index', metabolite]], left_on='metabolite_sub_id', right_on='index').drop(columns=['index'])
    print(metabolite)
    
    data['inoculation'] = pd.Categorical(data['inoculation'], categories=['Combo_15', 'Combo_1', 
                                                                          'Combo_10', 'Combo_13', 'Combo_14', 
                                                                          'Combo_16', 'Combo_2', 'Combo_4', 'Combo_7', 'Combo_9'], ordered=True)
    
    # Get media information (modify as per your data)
#     data['media'] = data['media_column']  # Replace 'media_column' with the actual column name from your data
    
    # Create formula for LMM
    formula = f'{metabolite} ~ C(inoculation)' 
    
    reference_level = 'Combo_15'

    # Update the formula with the specified reference level
    formula = f'{metabolite} ~ C(inoculation)' #, Treatment("{reference_level}"))
    vcf = { "sample_name": "0 + C(sample_name)"}                                                     
    # fit the GLM with sample_name as random effect
    try:
        mixed_model = mixedlm(formula , data=data, groups=data['sample_name'],
                               re_formula="~1")
        mixed_model_fit = mixed_model.fit(maxiter=1000)
        
        #extract relevant information from the model and compute p values
        coefs = mixed_model_fit.params
        std_err = mixed_model_fit.bse
        t_values = coefs / std_err
        df = mixed_model_fit.df_resid
        p_values = mixed_model_fit.pvalues.iloc[:-1].values # remove the groupvar output
        p_values # figure out a way to not include the groupvar

        # Calculate medians for different levels
        medians = list(data.groupby(['inoculation'])[metabolite].median())


        result_df = pd.DataFrame({'Probiotic_Combination': data['inoculation'].cat.categories,
                                          'coefficients': coefs.iloc[:-1], # i am ignoring the last row rightnow because that is the groupvars
                                           'pvalue': p_values,
                                          'median': medians,
                                          'metabolite':metabolite})
         # Append the results to the main DataFrame
        results_list.append(result_df)  # Append the DataFrame to the list
    except:
        continue

    print(mixed_model_fit.pvalues['Intercept'])
    print(mixed_model_fit.summary())

    
# Define a formatting function
def format_float(val):
    return '{:.4f}'.format(val)

results = pd.concat(results_list, ignore_index=True)
results['pvalue_corr'] = multipletests(results['pvalue'], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
results['pvalue_corr'] =  results[['pvalue_corr']].applymap(format_float)
results['pvalue_corr'] =results['pvalue_corr'].astype(float)
```
