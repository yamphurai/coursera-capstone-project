#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
customer = pd.read_csv(r"E:\Online Learning files\Machine Learning with Python\Capstone project\loan.csv")

X,y = make_blobs(n_samples = 10000, centers = [[0,2],[2,2],[1,1],[3,3]], cluster_std = 0.3)


k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12  )
k_means.fit(X)

#labelling each point in model
k_means_labels = k_means.labels_

#getting coordinates of the clusters centroids
k_means_cluster_centers = k_means.cluster_centers_

#create visual plot of the model
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[0,2], [2 , 2], [1, 1], [3, 3]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()



# In[ ]:




