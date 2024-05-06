import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("autodata.csv")
# datawrangling
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.notnull())
print(df.isnull())

df['normalized-losses'].fillna(df['normalized-losses'].mean(),inplace=True)
print(df.head())
# avg=df['normalized-losses'].astype("float").mean()
# print(avg)
# df['normalized-losses'].replace(np.nan,avg,inplace=True)

# datastandardization
df['city-L/100km']=235/df['city-mpg']
print(df.head())

# datatransformation
df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
print(df.head())

# binning

bins=3
group=['low','medium','high']
df['horsepower-binned']=pd.cut(df['horsepower'],bins,labels=group,include_lowest=True)
print(df['horsepower-binned'])

plt.hist(df['horsepower'])
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepowerbins")
plt.show()


