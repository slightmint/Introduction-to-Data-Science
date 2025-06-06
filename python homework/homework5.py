import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import time
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 设置为您想使用的核心数
os.environ["OMP_NUM_THREADS"] = "3"     # 避免KMeans内存泄漏
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体，根据您的系统可能需要调整
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据加载和预处理
def load_data():
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用PCA降维到2维以便可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_scaled, X_pca, y

# 手写实现K-均值聚类算法
class KMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        
    def fit(self, X):
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        # 迭代优化
        for _ in range(self.max_iter):
            # 分配样本到最近的聚类中心
            distances = self._calc_distances(X)
            self.labels_ = np.argmin(distances, axis=1)
            
            # 更新聚类中心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    new_centroids[k] = np.mean(X[self.labels_ == k], axis=0)
                else:
                    # 如果某个簇没有样本，随机选择一个样本作为中心
                    new_centroids[k] = X[np.random.randint(0, n_samples)]
            
            # 检查收敛性
            if np.sum((new_centroids - self.centroids) ** 2) < self.tol:
                break
                
            self.centroids = new_centroids
        
        return self
    
    def _calc_distances(self, X):
        # 计算每个样本到每个聚类中心的距离
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        return distances
    
    def predict(self, X):
        # 预测新样本的簇标签
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

# 手写实现AGNES层次聚类算法
class AGNES:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 初始化：每个样本作为一个簇
        clusters = [[i] for i in range(n_samples)]
        
        # 计算初始距离矩阵
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.sum((X[i] - X[j]) ** 2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # 合并簇直到达到指定的簇数量
        while len(clusters) > self.n_clusters:
            # 找到最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = 0, 0
            
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # 根据不同的连接方式计算簇间距离
                    if self.linkage == 'single':
                        # 单连接：最小距离
                        cluster_dist = float('inf')
                        for idx1 in clusters[i]:
                            for idx2 in clusters[j]:
                                if dist_matrix[idx1, idx2] < cluster_dist:
                                    cluster_dist = dist_matrix[idx1, idx2]
                    elif self.linkage == 'complete':
                        # 全连接：最大距离
                        cluster_dist = 0
                        for idx1 in clusters[i]:
                            for idx2 in clusters[j]:
                                if dist_matrix[idx1, idx2] > cluster_dist:
                                    cluster_dist = dist_matrix[idx1, idx2]
                    else:  # 'average'
                        # 平均连接：平均距离
                        cluster_dist = 0
                        count = 0
                        for idx1 in clusters[i]:
                            for idx2 in clusters[j]:
                                cluster_dist += dist_matrix[idx1, idx2]
                                count += 1
                        cluster_dist /= count
                    
                    if cluster_dist < min_dist:
                        min_dist = cluster_dist
                        merge_i, merge_j = i, j
            
            # 合并最近的两个簇
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # 生成标签
        self.labels_ = np.zeros(n_samples, dtype=int)
        for i, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = i
        
        return self

# 手写实现DBSCAN聚类算法
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # 初始化所有点为噪声点
        
        # 计算距离矩阵
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # 找出每个点的邻居
        neighbors = []
        for i in range(n_samples):
            neighbors.append(np.where(dist_matrix[i] <= self.eps)[0])
        
        # 聚类过程
        cluster_id = 0
        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue  # 已经被访问过
            
            if len(neighbors[i]) < self.min_samples:
                self.labels_[i] = -1  # 标记为噪声点
                continue
            
            # 开始一个新的聚类
            self.labels_[i] = cluster_id
            # 将邻居加入队列
            seed_queue = list(neighbors[i])
            
            # 扩展聚类
            j = 0
            while j < len(seed_queue):
                current_point = seed_queue[j]
                
                # 如果是噪声点，标记为当前簇
                if self.labels_[current_point] == -1:
                    self.labels_[current_point] = cluster_id
                
                # 如果未分类，标记为当前簇并检查其邻居
                if self.labels_[current_point] == -1:
                    self.labels_[current_point] = cluster_id
                    
                    # 如果是核心点，将其未访问的邻居加入队列
                    if len(neighbors[current_point]) >= self.min_samples:
                        for neighbor in neighbors[current_point]:
                            if neighbor not in seed_queue and self.labels_[neighbor] == -1:
                                seed_queue.append(neighbor)
                
                j += 1
            
            cluster_id += 1
        
        return self

# 评估聚类性能
def evaluate_clustering(y_true, y_pred):
    # 调整兰德指数（越接近1越好）
    ari = adjusted_rand_score(y_true, y_pred)
    
    # 轮廓系数（越接近1越好）
    try:
        silhouette = silhouette_score(X_pca, y_pred)
    except:
        silhouette = "无法计算（可能只有一个簇或存在噪声点）"
    
    return ari, silhouette

# 可视化聚类结果
def visualize_clustering(X_pca, y_true, y_pred, title):
    plt.figure(figsize=(12, 5))
    
    # 真实标签
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=30)
    plt.title('真实标签')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    
    # 聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    
    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == "__main__":
    # 加载数据
    X_scaled, X_pca, y_true = load_data()
    
    print("乳腺癌数据集聚类分析")
    print("="*50)
    
    # 1. 手写K-均值聚类
    print("\n1. 手写K-均值聚类")
    start_time = time.time()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_scaled)
    my_kmeans_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, kmeans.labels_)
    print(f"运行时间: {my_kmeans_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, kmeans.labels_, "手写K-均值聚类结果")
    
    # 2. 手写AGNES聚类
    print("\n2. 手写AGNES聚类")
    start_time = time.time()
    agnes = AGNES(n_clusters=2, linkage='average')
    agnes.fit(X_scaled)
    my_agnes_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, agnes.labels_)
    print(f"运行时间: {my_agnes_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, agnes.labels_, "手写AGNES聚类结果")
    
    # 3. 手写DBSCAN聚类
    print("\n3. 手写DBSCAN聚类")
    start_time = time.time()
    dbscan = DBSCAN(eps=3.0, min_samples=5)  # 参数需要根据数据调整
    dbscan.fit(X_scaled)
    my_dbscan_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, dbscan.labels_)
    print(f"运行时间: {my_dbscan_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    if isinstance(silhouette, str):
        print(f"轮廓系数: {silhouette}")
    else:
        print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, dbscan.labels_, "手写DBSCAN聚类结果")
    
    # 4. 使用sklearn库进行对比
    print("\n4. 使用sklearn库进行对比")
    
    # sklearn K-means
    from sklearn.cluster import KMeans as SKLearnKMeans
    start_time = time.time()
    sk_kmeans = SKLearnKMeans(n_clusters=2, random_state=42)
    sk_kmeans.fit(X_scaled)
    sk_kmeans_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, sk_kmeans.labels_)
    print("\nScikit-learn K-means:")
    print(f"运行时间: {sk_kmeans_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, sk_kmeans.labels_, "Sklearn K-means聚类结果")
    
    # sklearn AgglomerativeClustering (AGNES)
    from sklearn.cluster import AgglomerativeClustering
    start_time = time.time()
    sk_agnes = AgglomerativeClustering(n_clusters=2)
    sk_agnes.fit(X_scaled)
    sk_agnes_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, sk_agnes.labels_)
    print("\nScikit-learn AgglomerativeClustering:")
    print(f"运行时间: {sk_agnes_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, sk_agnes.labels_, "Sklearn AGNES聚类结果")
    
    # sklearn DBSCAN
    from sklearn.cluster import DBSCAN as SKLearnDBSCAN
    start_time = time.time()
    sk_dbscan = SKLearnDBSCAN(eps=3.0, min_samples=5)
    sk_dbscan.fit(X_scaled)
    sk_dbscan_time = time.time() - start_time
    
    ari, silhouette = evaluate_clustering(y_true, sk_dbscan.labels_)
    print("\nScikit-learn DBSCAN:")
    print(f"运行时间: {sk_dbscan_time:.4f}秒")
    print(f"调整兰德指数: {ari:.4f}")
    if isinstance(silhouette, str):
        print(f"轮廓系数: {silhouette}")
    else:
        print(f"轮廓系数: {silhouette:.4f}")
    
    visualize_clustering(X_pca, y_true, sk_dbscan.labels_, "Sklearn DBSCAN聚类结果")
    
    # 性能对比
    print("\n5. 性能对比")
    print("="*50)
    print("算法\t\t手写实现时间(秒)\tSklearn实现时间(秒)\t加速比")
    print(f"K-means\t\t{my_kmeans_time:.4f}\t\t{sk_kmeans_time:.4f}\t\t{my_kmeans_time/sk_kmeans_time:.2f}")
    print(f"AGNES\t\t{my_agnes_time:.4f}\t\t{sk_agnes_time:.4f}\t\t{my_agnes_time/sk_agnes_time:.2f}")
    print(f"DBSCAN\t\t{my_dbscan_time:.4f}\t\t{sk_dbscan_time:.4f}\t\t{my_dbscan_time/sk_dbscan_time:.2f}")