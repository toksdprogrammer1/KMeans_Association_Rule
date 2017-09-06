import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import pandas as pd
import numpy as np
from mlxtend.cluster import Kmeans

df = pd.read_excel("data.xlsx", sheetname=0)
df2 = pd.read_excel("data.xlsx", sheetname=2)

print(df2)

X = df.values
y = df2.values

fig = plt2.figure("Before Clustering", figsize=(5, 5))

plt2.scatter(X[:, 1], X[:, 3], c='black')
#plt2.show()

km = Kmeans(k=3,
            max_iter=50,
            random_seed=1,
            print_progress=3)

km.fit(X)

print('Iterations until convergence:', km.iterations_)
print('Final centroids:\n', km.centroids_)

y_clust = km.predict(X)
df['LIFESTYLE_CHANGES'] = y
df['Model Test'] = y_clust
#new = pd.merge(df, df2)
#print(df)


fig = plt3.figure("Color", figsize=(5, 5))

y = y.flatten()
plt3.scatter(X[y == 0, 1],
            X[y == 0, 3],
            s=50,
            c='lightgreen',

            label='Bad')

plt3.scatter(X[y == 1,1],
            X[y == 1,3],
            s=50,
            c='orange',

            label='Fair')

plt3.scatter(X[y == 2,1],
            X[y == 2,3],
            s=50,
            c='lightblue',

            label='Good')

plt3.legend(loc='lower left',
           scatterpoints=1)

fig = plt.figure("KMeans Clusters", figsize=(5, 5))


plt.scatter(X[y_clust == 0, 1],
            X[y_clust == 0, 3],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')

plt.scatter(X[y_clust == 1,1],
            X[y_clust == 1,3],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')

plt.scatter(X[y_clust == 2,1],
            X[y_clust == 2,3],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')


plt.scatter(km.centroids_[:,1],
            km.centroids_[:,3],
            s=250,
            marker='*',
            c='red',
            label='centroids')

plt.legend(loc='lower left',
           scatterpoints=1)
plt.grid()
plt.show()

writer = pd.ExcelWriter('output.xlsx')
df.to_excel(writer,'Sheet1')
#writer.save()

def encode_units(x):
    if x == 0:
        return 'Bad'
    elif x == 1:
        return 'Fair'
    else:
        return 'Good'

df['LIFESTYLE_CHANGES'] = df['LIFESTYLE_CHANGES'].apply(encode_units)
df['Model Test'] = df['Model Test'].apply(encode_units)
df['Match'] = np.where(df['LIFESTYLE_CHANGES'] == df['Model Test'], 'Yes', 'No')

#print(df)
print(df[ (df['Match'] == 'Yes')].count())
df.to_excel(writer,'Sheet2')
writer.save()