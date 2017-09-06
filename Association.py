import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import OnehotTransactions

from sklearn import datasets

# Read the excel sheet to pandas dataframe
df = pd.read_excel("data.xlsx", sheetname=0)
df2 = pd.read_excel("data.xlsx", sheetname=2)
df3 = pd.read_excel("output.xlsx", sheetname=0)
X = df #iris.data
X['LIFESTYLE_CHANGES'] = df2
#print(X)

def encode_units1(x):
    if x > 2 and x <= 5:
        return 'RISK: 2 - 5'
    elif x > 6 and x <= 9:
        return 'RISK: 6 - 9'
    else:
        return 'RISK: greater than 9'

def encode_units0(x):
    if x >= 17 and x < 35:
        return 'AGE: 17 - 34'
    elif x >=35 and x <= 65:
        return 'AGE: 35 - 65'
    else:
        return 'AGE: greater than 65'

def encode_units2(x):
    if x >= 1 and x <= 1.5:
        return 'BP: 1 - 1.5'
    elif x >= 1.6 and x <= 2:
        return 'BP: 1.6 - 2'
    else:
        return 'BP: greater than 2'

def encode_units3(x):
    if x >= 0 and x <= 30:
        return 'WEIGHT: 0 - 30'
    elif x >= 31 and x <= 80:
        return 'WEIGHT: 31 - 80'
    else:
        return 'WEIGHT: greater than 80'


X = df3[ (df3['LIFESTYLE_CHANGES'] == 2)]

# writer = pd.ExcelWriter('cluster1.xlsx')
# X.to_excel(writer,'Sheet0')
# writer.save()

X.iloc[:,0] = X.iloc[:,0].apply(encode_units0)
X.iloc[:,1] = X.iloc[:,1].apply(encode_units1)
X.iloc[:,2] = X.iloc[:,2].apply(encode_units2)
X.iloc[:,3] = X.iloc[:,3].apply(encode_units3)
del X['LIFESTYLE_CHANGES']
del X['Model Test']
# writer = pd.ExcelWriter('converted-cluster3.xlsx')
# X.to_excel(writer,'Sheet0')
# writer.save()
y = X.values
#print(X)
oht = OnehotTransactions()
oht_ary = oht.fit(y).transform(y)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
#print(df)
frequent_itemsets = apriori(df, min_support=0.07, use_colnames=True)
#print(frequent_itemsets)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
writer = pd.ExcelWriter('cluster3-rules.xlsx')
rules.to_excel(writer,'Sheet0')
writer.save()

filtered = rules[ (rules['confidence'] >= 0.5) ]
writer = pd.ExcelWriter('filtered-rules.xlsx')
filtered.to_excel(writer,'Sheet0')
writer.save()

print(rules[ (rules['confidence'] >= 0.5) ])