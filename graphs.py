import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset/weather.csv")

sns.pairplot(data)
plt.show()

sns.heatmap(data.corr(),annot=True)
plt.show()