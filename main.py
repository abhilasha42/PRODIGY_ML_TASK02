import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
data = pd.read_csv("Task2/data/Mall_Customers.csv")

print("Dataset loaded successfully")
print(data.head())

# using annual income and spend score for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# use of elbow method
wcss = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# graph plotting
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method to Find Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# applying Kmeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# adding cluster label to the datset
data['Cluster'] = clusters
print("Clustering completed")

# visualizing cluster
plt.scatter(
    features['Annual Income (k$)'],
    features['Spending Score (1-100)'],
    c=clusters
)

# centroids plotting
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()
print("Task 2 completed successfully")