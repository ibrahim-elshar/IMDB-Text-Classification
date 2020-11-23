# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:31:16 2019

@author: IJE8
"""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


classifier = [
'Baseline: Bi-LSTM',
'Baseline: NB',
'Logistic Regression',
'Decision Tree',
'Random Forest',
'SVD(RBF Kernel)',
'SVD(Linear Kernel)',
'Gaussian NB',
'K-NN(k=1)',
'K-NN(k=5)',
'K-NN(k=10)',
'AdaBoost',
'Radius-NN(r=10)',
'Paragraph Embeddings',
'End to End Embeddings' 
]
Acc = np.array([  [0.8710],
                  [0.8475],
                  [0.8496],
                  [0.8714],
                  [0.8708],
                  [0.8834],
                  [0.8537],
                  [0.8518],
                  [0.8779],
                  [0.8830],
                  [0.8796],
                  [0.8606],
                  [0.8460],
                  [0.8850],
                  [0.9130]])


df = pd.DataFrame()
df = pd.DataFrame(columns=['Accuracy','Classifier'])
df['Accuracy'] = Acc.ravel()*100
df['Classifier'] = classifier

sns.set(style="whitegrid")
#tips = sns.load_dataset("tips")
#ax = sns.barplot(x="day", y="total_bill", data=tips)
sns.set(font_scale=2)
groupedvalues=df.groupby('Accuracy').sum().reset_index()
pal = sns.color_palette("Greens_d", len(groupedvalues))
rank = groupedvalues["Accuracy"].argsort().argsort() 
g = sns.barplot(x='Accuracy', y='Classifier',data=groupedvalues, palette=np.array(pal[::-1])[rank]) #data=df)
for index, row in groupedvalues.iterrows():
    g.text(row.Accuracy,row.name, round(row.Accuracy,4), color='black', ha="left")
plt.xlabel("Test Accuracy %")
plt.ylabel("Classifier")
plt.show()

##import seaborn as sns
##import matplotlib.pyplot as plt
##import numpy as np
#
##df = sns.load_dataset("tips")
#groupedvalues=df.groupby('Classifier').sum().reset_index()
#
##pal = sns.color_palette("Greens_d", len(groupedvalues))
#rank = groupedvalues["total_bill"].argsort().argsort() 
#g=sns.barplot(x='day',y='tip',data=groupedvalues, palette=np.array(pal[::-1])[rank])
#
#for index, row in groupedvalues.iterrows():
#    print(row.name,row.tip, round(row.total_bill,2))
#    g.text(row.name,row.tip, round(row.total_bill,2), color='black', ha="center")
#
#plt.show()
#
#
#import seaborn as sns
#import matplotlib.pyplot as plt
#import numpy as np
#
#df = sns.load_dataset("tips")
#groupedvalues=df.groupby('day').sum().reset_index()
#
#pal = sns.color_palette("Greens_d", len(groupedvalues))
#rank = groupedvalues["total_bill"].argsort().argsort() 
#g=sns.barplot(x='day',y='tip',data=groupedvalues, palette=np.array(pal[::-1])[rank])
#
#for index, row in groupedvalues.iterrows():
#    print(row.name,row.tip, round(row.total_bill,2))
#    g.text(row.name,row.tip, round(row.total_bill,2), color='black', ha="center")
#
#plt.show()