import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/titanic_data.csv')
print(df)
print(df.describe(include='object'))

# finding patterns :
sns.countplot(df['Survived'])
sns.boxplot(df['Fare'])
sns.catplot(x='Pclass',y='Age',data=df,kind='box')
sns.pairplot(df)
sns.scatterplot(x='Fare',y='Pclass',hue='Survived',data=df)
sns.distplot(df['Fare'])

#fare ticket price
sns.catplot(x='Pclass',y='Fare',data=df,kind='bar')
plt.show()