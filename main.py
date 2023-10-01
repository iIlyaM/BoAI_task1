import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS


def main():
    data = pd.read_csv('files/customers.csv', delimiter=';')

    distance_matrix = linkage(data, method='ward', metric='euclidean')
    plt.figure(figsize=(10, 5))
    dendrogram(distance_matrix, p=3, truncate_mode='level')
    plt.title('Дендограмма')
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_

    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=0)
    X_mds = mds.fit_transform(data)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=labels)
    plt.show()

    wcss = []  # Within-Cluster-Sum-of-Squares

    for i in range(1, 11):  # Пробуем разное количество кластеров
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Выведем график "каменистая осыпь"
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('График "каменистая осыпь"')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма квадратов внутри кластера')
    plt.show()

    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
    kmeans.fit(data)
    data['cluster'] = kmeans.labels_

    # Средние значения переменных для каждого кластера
    print(data.groupby('cluster').mean())


if __name__ == '__main__':
    main()
