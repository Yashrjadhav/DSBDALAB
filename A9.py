import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv")
print(df)

#replacing all null values by means
df['Age']=df['Age'].fillna(np.mean(df['Age']))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=df, palette='Set2')
plt.title('Distribution of age wrt to gender')
plt.show()

