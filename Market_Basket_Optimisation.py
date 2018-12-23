# Importing dataset
import pandas as pd
dataset=pd.read_csv("F://ML//Machine Learning A-Z Template Folder\Part 5 - Association Rule Learning//Section 28 - Apriori//Apriori-Python//Apriori_Python//Market_Basket_Optimisation.csv",header=None)

# Data Preprocessing
transactions=[]
for i in range(0,len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

# Training 
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_length=2,min_lift=3)    

# visualizing
results=list(rules)

import matplotlib.pyplot as plt
plt.itemFrequency(results)
