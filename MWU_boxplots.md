
The Mann-Whitney U test is a non-parametric test used to compare differences between two independent groups when the data are not normally distributed. It assesses whether the distribution of ranks differs significantly between the groups.

Boxplots, or box-and-whisker plots, visually summarize the distribution of a dataset, displaying the median, quartiles, and potential outliers, making it easy to compare distributions between groups.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import combinations

# Assuming meta_dmm is properly defined elsewhere
# Example variables
var1 = 'dmm_cluster'
var2 = 'species_richness'

# Selecting and scaling the relevant data
_scaled = meta_dmm[[var1, var2]]

# Define the category order based on unique values in var1
category_order = sorted(_scaled[var1].unique())

# Set plot style (optional)
sns.set(style='whitegrid')

# Create a boxplot with swarmplot overlay
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x=var1, y=var2, data=_scaled, palette='Set1', order=category_order)  # Adjust palette as desired
sns.stripplot(x=var1, y=var2, data=_scaled, color='black', size=5, alpha=0.5, order=category_order)

# Perform Mann-Whitney U test for each pair of categories and annotate the plot
pairs = list(combinations(category_order, 2))
annotations = []

for (cat1, cat2) in pairs:
    group1 = _scaled[_scaled[var1] == cat1][var2]
    group2 = _scaled[_scaled[var1] == cat2][var2]
    statistic, p_value = mannwhitneyu(group1, group2)
    annotations.append((cat1, cat2, p_value))

# Add annotations to the plot
y_max = _scaled[var2].max()
y_range = _scaled[var2].max() - _scaled[var2].min()

for i, (cat1, cat2, p_value) in enumerate(annotations):
    x1, x2 = category_order.index(cat1), category_order.index(cat2)
    y, h, col = y_max + (i+1) * 0.05 * y_range, 0.02 * y_range, 'k'
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    ax.text((x1 + x2) * 0.5, y + h, f'p = {p_value:.3e}', ha='center', va='bottom', color=col, fontsize=12)

# Annotate the plot with the Mann-Whitney U test result
plt.title(f'Boxplot of {var2} by {var1}', fontsize=16)
plt.xlabel(var1, fontsize=14)
plt.ylabel(var2, fontsize=14)

# Show the plot
plt.tight_layout()
plt.show()
```
