
Pearson correlation measures the strength and direction of the linear relationship between two continuous variables, assuming normally distributed data. Spearman correlation, a non-parametric test, measures the strength and direction of the monotonic relationship between two variables using their rank orders, making it suitable for non-normal data. Scatter plots visually represent the relationship between two variables, with data points plotted on a Cartesian plane, helping to identify trends, patterns, and potential correlations. Together, these methods provide a comprehensive view of the relationships between variables in a dataset.

```python


# sns.scatterplot(x= 'species_richness', y='BMI (calc)', data = meta)

# Create a mapping for education levels
var1='Diet Score'
var2 = 'vf_richness'
# Calculate Pearson correlation coefficient and p-value

scaled_data = (meta_dmm_vf[[var1,var2]])
_scaled = pd.DataFrame(scaled_data, columns=[var1,var2])
corr_coef, p_value = spearmanr(_scaled[var1], _scaled[var2])

# Set plot style
sns.set(style='whitegrid')

# Create a scatter plot with regression line
plt.figure(figsize=(14, 9))
scatter_plot = sns.regplot(x=var1, y=var2, data=_scaled, ci=None, scatter_kws={'s': 100}, line_kws={"color": "red"})

# Set the x-ticks to use the edu_mapping
# plt.xticks(ticks=list(edu_mapping.keys()), labels=list(edu_mapping.values()), rotation=45, ha='right')

# Annotate the plot with the correlation coefficient and p-value
plt.title(f'Scatter Plot of {var1} vs  {var2}', fontsize=16)
plt.xlabel(var1, fontsize=14)
plt.ylabel(var2, fontsize=14)
plt.text(1, max(_scaled[var2]) - 5, f'Spearman r: {corr_coef:.2f}\nP-value: {p_value:.2g}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Customize the background
scatter_plot.figure.set_facecolor('white')

# Show the plot
plt.show()
```
