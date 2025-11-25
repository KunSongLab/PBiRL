import numpy as np
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_json("../outputs/completed_vehicle_info.json")
temp = data.groupby("vehicle_id").apply(lambda x: np.sum(x["unit"] * x["travel_time"]) / np.sum(x["unit"]))
data = data.groupby("vehicle_id").mean()
data["travel_time"] = temp
print(data)

# sns.barplot(data=data, x=data.index, y="travel_time")
plt.bar(data.index, data["travel_time"])
plt.show()