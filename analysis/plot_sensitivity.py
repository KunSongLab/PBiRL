import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filename = "../controls/run/toy_small2/results/sensitivity.csv"
data = pd.read_csv(filename)
print(data)
plt.figure(figsize=[5, 4])
g = sns.lineplot(data=data, x="Ratio", y="TTT", hue="Group", style="Group", linewidth=3)
# g = sns.lineplot(data=data, x="Ratio", y="Delay rate", hue="Group", style="Group", linewidth=3)
# g = sns.lineplot(data=data, x="Ratio", y="Speed (km/h)", hue="Group", style="Group", linewidth=3)
g.legend_.set_title(None)
plt.ylabel("Total travel time", fontsize=15)
plt.xlabel("Demand ratio", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.ylim(-2000, 1500)
plt.legend(fontsize=15)
plt.show()