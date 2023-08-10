import pandas as pd
from scipy.stats import kruskal

# Assume you have the coverage data for each model in separate lists
model1_coverage = [1000000, 1500000, 12]
model2_coverage = [110, 40, 10]
model3_coverage = [90, 160, 115]
model4_coverage = [105, 135, 125]

# Create a DataFrame to organize the data
data = pd.DataFrame({
    'Model 1': model1_coverage,
    'Model 2': model2_coverage,
    'Model 3': model3_coverage,
    'Model 4': model4_coverage
})

# Perform the Kruskal-Wallis test
statistic, p_value = kruskal(data['Model 1'], data['Model 2'], data['Model 3'], data['Model 4'])

# Interpret the results based on the p-value
print(statistic,p_value )
if p_value < 0.05:
    print("There is a significant difference in coverage among the models.")
else:
    print("There is no significant difference in coverage among the models.")
