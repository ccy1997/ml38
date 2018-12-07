'''
Created on 1 Dec 2018

@author: User
'''
import seaborn as sns
import statistics
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from idlelib.textview import view_file
from sklearn.covariance.elliptic_envelope import EllipticEnvelope
pd.set_option('display.max_columns', None)
df= pd.read_csv(r"C:\Users\User\Desktop\housing.csv", dialect='excel',sep=',')


plt.figure(figsize=(10,4))
plt.hist(df[df['total_bedrooms'].notnull()]['total_bedrooms'],bins=20,color='green')#histogram of totalbedrooms
#data has some outliers
(df['total_bedrooms']>4000).sum()
plt.title('frequency histogram')
plt.xlabel('total bedrooms')
plt.ylabel('frequency')
plt.show()

print(df.describe())
sns.pairplot(df)
plt.figure(figsize=(10,6))
sns.distplot(df['median_house_value'],color='purple') 

plt.figure(figsize=(10,6))

plt.scatter(df['population'],df['median_house_value'],c=df['median_house_value'],s=df['median_income']*50)
plt.colorbar
plt.title('population/house value' )
plt.xlabel('population')
plt.ylabel('house value')
plt.plot()

df[df['median_house_value']>450000]['median_house_value'].value_counts().head()
df=df.loc[df['median_house_value']<500001,:]
df=df[df['population']<25000]
plt.figure(figsize=(10,6))
sns.distplot(df['median_house_value'])
plt.show()

def getOutliers(dataframe,column):
    column = "total_rooms" 
    df[column].plot.box(figsize=(8,8))
    des = dataframe[column].describe()
    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
    Q1 = des[desPairs['25']]
    Q3 = des[desPairs['75']]
    IQR = Q3-Q1
    lowerBound = Q1-1.5*IQR
    upperBound = Q3+1.5*IQR
    print("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
    
    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]

    print("Outliers out of total = {} are \n {}".format(df[column].size,len(data[column])))
    
    outlierRemoved = df[column].isin(data[column])
    return outlierRemoved

df_outliersRemoved = getOutliers(df,"total_rooms")

plt.figure(figsize=(15,10))
plt.scatter(df['longitude'],df['latitude'],c=df['median_house_value'],s=df['population']/10,cmap='viridis')
plt.colorbar()
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('house price/geo-coordinates')
plt.show()

#corelation matrix
plt.figure(figsize=(11,7))
sns.heatmap(cbar=False,annot=True,data=df.corr()*100,cmap='coolwarm')
plt.title('% Corelation Matrix')
plt.show()

#barplot on ocean_proximity
plt.figure(figsize=(10,6))
sns.countplot(data=df,x='ocean_proximity')
plt.plot()

#boxplot of house value on ocean_proximity categories
plt.figure(figsize=(10,6))
sns.boxplot(data=df,x='ocean_proximity',y='median_house_value',palette='viridis')
plt.plot()


