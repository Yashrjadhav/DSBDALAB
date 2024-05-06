import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import datasets

iris=datasets.load_iris()
df=pd.DataFrame(iris['data'])
df.rename(columns={0:'sepallength',1:'sepalwidth',2:'petallength',3:'petalwidth',4:'species'},inplace=True)
#displaying the datatypes
print(df.dtypes)

#displaying outliers and tabular data
sns.histplot(x=df['sepallength'],kde=True) #do it for every column
sns.boxplot(df['sepalwidth'])  #do it for every column

sns.boxplot(x='sepallength',y='sepalwidth',data=df)  #here instead of sepalwidth use species but as we are taking iris dataset from the sklearn where there is no species column (if we want we would need to create it)
plt.show()