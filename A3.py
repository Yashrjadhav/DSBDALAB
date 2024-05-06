import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
iris=datasets.load_iris()
df=pd.DataFrame(iris['data'])
# provides summary of dataset
print(df.describe())

#renaming column names
df.rename(columns={0:'sepallength',1:'sepalwidth',2:'petallength',3:'petalwidth',4:'species'},inplace=True)
print(df.head())
print("mean",df.mean())
print("median",df.median())
print("min",df.min())
print("max",df.max())
# print("mode",df['species'].mode())

df.groupby(df['species']).count()

#displaying standard deviation for each column
print(df.sepallength.std())