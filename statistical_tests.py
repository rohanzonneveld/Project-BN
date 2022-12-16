import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np


# Full data
data = pd.read_csv('results_dsep.csv')
x = data.iloc[:,2]
y = data.iloc[:,3]

coeff, pvalue = pearsonr(x,y)
print(coeff, pvalue)

numrows = data.shape[0]

# Excluded datapoints with less than 50 nodes and edges

# Test variance between equal queries
percentages = []
nodes_edges = []
for i in range(0,numrows-2,3):
    var = 0
    for j in range(3):
        var += data.iloc[i+j,3]
    mean = int(var/3)
    for k in range(3):
        if mean != 0:
            diff = int(data.iloc[i+k,3] - mean)
            perc = diff/mean
        else:
            perc = 0
        percentages.append(perc*100)
        nodes_edges.append(data.iloc[i+k,2])

first_quartile = np.percentile(percentages, 25)
third_quartile = np.percentile(percentages, 75)
max_y = max(percentages)+1

plt.scatter(nodes_edges,percentages)
# plt.boxplot(percentages, whis=([0,83]))
plt.xlabel('Sum of nodes and edges')
plt.ylabel('Percentage difference')
plt.title('Percentage difference of runtime on equal queries and networks')
plt.show()


