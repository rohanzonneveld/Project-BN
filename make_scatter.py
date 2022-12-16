import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results_dsep.csv', header=0)

# Select the columns you want to plot by their index
# In this example, we're plotting the first and second columns
x = data.iloc[:, 2]
y = data.iloc[:, 3]

# Create the scatter plot
plt.scatter(x, y)
plt.xlabel("Sum of nodes and edges")
plt.ylabel('Runtime [s]')
plt.title('Development of runtime with increasing network size')

# Show the plot
plt.show()




