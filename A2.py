import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("autodata2.csv")

print(df.head())
df['wheel-base'].fillna(df['wheel-base'].mean(), inplace=True)
print(df.head())


#displaying outliers and resolving it 
from scipy import stats
z_scores = np.abs(stats.zscore(df['normalized-losses']))
outliers_indices = np.where(z_scores > 3)
df = df.drop(outliers_indices[0])
print(outliers_indices[0])
print(z_scores)

#Tranformations -Log transformation 
df['price']=np.log(df['price'])
print(df)


#displalying outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['horsepower'])
plt.xlabel("Horsepower")
plt.title("Box Plot of Horsepower")
plt.show()
