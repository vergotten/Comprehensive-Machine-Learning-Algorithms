# K-Means / K-средних

K-Means is a type of unsupervised learning algorithm which is used for clustering problem. It's a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.

K-средних - это тип алгоритма обучения без учителя, который используется для задачи кластеризации. Это алгоритм, основанный на центроидах, или алгоритм, основанный на расстояниях, где мы вычисляем расстояния, чтобы назначить точку кластеру. В K-средних каждый кластер связан с центроидом.

# Hierarchical Clustering / Иерархическая кластеризация

Hierarchical Clustering is another unsupervised learning algorithm that is used to group together the unlabeled data points having similar characteristics. Hierarchical clustering algorithms falls into following two categories.
- Agglomerative hierarchical algorithms − In agglomerative hierarchical algorithms, each data point is treated as a single cluster and then successively merge or agglomerate (bottom-up approach) the pairs of clusters. The hierarchy of the clusters is represented as a dendrogram or tree structure.
- Divisive hierarchical algorithms − On the other hand, in divisive hierarchical algorithms, all the data points are treated as one big cluster and the process of clustering involves dividing (Top-down approach) the one big cluster into various small clusters.

Иерархическая кластеризация - это еще один алгоритм обучения без учителя, который используется для группировки неразмеченных точек данных, имеющих схожие характеристики. Алгоритмы иерархической кластеризации делятся на следующие две категории.
- Агломеративные иерархические алгоритмы - в агломеративных иерархических алгоритмах каждая точка данных рассматривается как отдельный кластер, а затем последовательно объединяют или агломерируют (снизу вверх) пары кластеров. Иерархия кластеров представлена в виде дендрограммы или древовидной структуры.
- Дивизивные иерархические алгоритмы - с другой стороны, в дивизивных иерархических алгоритмах все точки данных рассматриваются как один большой кластер, и процесс кластеризации включает разделение (сверху вниз) одного большого кластера на различные малые кластеры.

# DBSCAN / DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clusering algorithm, which can discover clusters of different shapes and sizes from a large amount of data, which is containing noise and outliers. The main idea is that a point belongs to a cluster if it is close to many points from that cluster.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) - это алгоритм кластеризации на основе плотности, который может обнаруживать кластеры различной формы и размера из большого количества данных, содержащих шум и выбросы. Основная идея заключается в том, что точка принадлежит кластеру, если она близка к многим точкам из этого кластера.