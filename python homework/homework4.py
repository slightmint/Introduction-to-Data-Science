import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体，根据您的系统可能需要调整
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载乳腺癌数据集
def load_data():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    target_names = cancer.target_names
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test, feature_names, target_names

# 评估模型性能
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{model_name} 性能评估:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# 可视化混淆矩阵
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} 混淆矩阵')
    plt.colorbar()
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks([0, 1], ['良性', '恶性'])
    plt.yticks([0, 1], ['良性', '恶性'])
    
    # 在混淆矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

# 1. K-近邻算法实现
class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # 计算欧氏距离
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # 获取k个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            # 获取这些邻居的标签
            k_nearest_labels = self.y_train[k_indices]
            # 投票决定预测标签
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        
        return np.array(predictions)

# 2. 朴素贝叶斯算法实现
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 初始化均值、方差和先验概率
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # 计算每个类别的均值、方差和先验概率
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        posteriors = []
        
        # 计算每个类别的后验概率
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        # 返回具有最高后验概率的类别
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # 防止除以零
        var = np.where(var < 1e-10, 1e-10, var)
        
        # 计算高斯概率密度函数
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# 3. 逻辑回归算法实现
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# 4. 决策树算法实现
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # 特征索引
            self.threshold = threshold  # 分割阈值
            self.left = left  # 左子树
            self.right = right  # 右子树
            self.value = value  # 叶节点的值
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)
        
        # 寻找最佳分割
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # 创建子树
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return self.Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, n_features):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # 对每个特征
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            # 对每个阈值
            for threshold in thresholds:
                # 计算信息增益
                gain = self._information_gain(X[:, feature], y, threshold)
                
                # 更新最佳分割
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, X_feature, y, threshold):
        # 父节点的熵
        parent_entropy = self._entropy(y)
        
        # 创建左右子节点
        left_indices = X_feature <= threshold
        right_indices = ~left_indices
        
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        
        # 计算左右子节点的熵
        n = len(y)
        n_l, n_r = len(y[left_indices]), len(y[right_indices])
        e_l, e_r = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        
        # 计算信息增益
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        # 计算熵
        class_labels = np.unique(y)
        entropy = 0
        
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        
        return entropy
    
    def _most_common_label(self, y):
        # 返回最常见的标签
        return np.bincount(y).argmax()
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        # 如果是叶节点
        if node.value is not None:
            return node.value
        
        # 决定走左子树还是右子树
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# 5. 支持向量机算法实现
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        # 将标签转换为{-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.where(linear_output <= 0, 0, 1)

# 使用sklearn库进行对比实验
def sklearn_comparison(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression as SklearnLR
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    
    # KNN
    start_time = time.time()
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)
    knn_sklearn.fit(X_train, y_train)
    y_pred = knn_sklearn.predict(X_test)
    knn_time = time.time() - start_time
    print(f"\nScikit-learn KNN 执行时间: {knn_time:.4f}秒")
    evaluate_model(y_test, y_pred, "Scikit-learn KNN")
    
    # 朴素贝叶斯
    start_time = time.time()
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    y_pred = nb_sklearn.predict(X_test)
    nb_time = time.time() - start_time
    print(f"\nScikit-learn 朴素贝叶斯 执行时间: {nb_time:.4f}秒")
    evaluate_model(y_test, y_pred, "Scikit-learn 朴素贝叶斯")
    
    # 逻辑回归
    start_time = time.time()
    lr_sklearn = SklearnLR(max_iter=1000)
    lr_sklearn.fit(X_train, y_train)
    y_pred = lr_sklearn.predict(X_test)
    lr_time = time.time() - start_time
    print(f"\nScikit-learn 逻辑回归 执行时间: {lr_time:.4f}秒")
    evaluate_model(y_test, y_pred, "Scikit-learn 逻辑回归")
    
    # 决策树
    start_time = time.time()
    dt_sklearn = DecisionTreeClassifier(max_depth=10)
    dt_sklearn.fit(X_train, y_train)
    y_pred = dt_sklearn.predict(X_test)
    dt_time = time.time() - start_time
    print(f"\nScikit-learn 决策树 执行时间: {dt_time:.4f}秒")
    evaluate_model(y_test, y_pred, "Scikit-learn 决策树")
    
    # 支持向量机
    start_time = time.time()
    svm_sklearn = SVC()
    svm_sklearn.fit(X_train, y_train)
    y_pred = svm_sklearn.predict(X_test)
    svm_time = time.time() - start_time
    print(f"\nScikit-learn 支持向量机 执行时间: {svm_time:.4f}秒")
    evaluate_model(y_test, y_pred, "Scikit-learn 支持向量机")

# 主函数
def main():
    print("加载乳腺癌数据集...")
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    print(f"数据集大小: {len(X_train) + len(X_test)} 样本")
    print(f"特征数量: {len(feature_names)}")
    print(f"类别: {target_names}")
    
    # 1. KNN算法
    print("\n正在运行自实现的KNN算法...")
    start_time = time.time()
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_time = time.time() - start_time
    print(f"KNN 执行时间: {knn_time:.4f}秒")
    evaluate_model(y_test, y_pred, "KNN")
    plot_confusion_matrix(y_test, y_pred, "KNN")
    
    # 2. 朴素贝叶斯算法
    print("\n正在运行自实现的朴素贝叶斯算法...")
    start_time = time.time()
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    nb_time = time.time() - start_time
    print(f"朴素贝叶斯 执行时间: {nb_time:.4f}秒")
    evaluate_model(y_test, y_pred, "朴素贝叶斯")
    plot_confusion_matrix(y_test, y_pred, "朴素贝叶斯")
    
    # 3. 逻辑回归算法
    print("\n正在运行自实现的逻辑回归算法...")
    start_time = time.time()
    lr = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr_time = time.time() - start_time
    print(f"逻辑回归 执行时间: {lr_time:.4f}秒")
    evaluate_model(y_test, y_pred, "逻辑回归")
    plot_confusion_matrix(y_test, y_pred, "逻辑回归")
    
    # 4. 决策树算法
    print("\n正在运行自实现的决策树算法...")
    start_time = time.time()
    dt = DecisionTree(max_depth=10)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    dt_time = time.time() - start_time
    print(f"决策树 执行时间: {dt_time:.4f}秒")
    evaluate_model(y_test, y_pred, "决策树")
    plot_confusion_matrix(y_test, y_pred, "决策树")
    
    # 5. 支持向量机算法实现
    print("\n正在运行自实现的支持向量机算法...")
    start_time = time.time()
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_time = time.time() - start_time
    print(f"支持向量机 执行时间: {svm_time:.4f}秒")
    evaluate_model(y_test, y_pred, "支持向量机")
    plot_confusion_matrix(y_test, y_pred, "支持向量机")
    
    # 使用sklearn库进行对比
    print("\n使用Scikit-learn库进行对比实验...")
    sklearn_comparison(X_train, X_test, y_train, y_test)
    
    # 绘制性能对比图
    print("\n绘制性能对比图...")
    models = ["KNN", "朴素贝叶斯", "逻辑回归", "决策树", "支持向量机"]
    custom_times = [knn_time, nb_time, lr_time, dt_time, svm_time]
    
    plt.figure(figsize=(12, 6))
    plt.bar(models, custom_times, color='blue', alpha=0.7, label='自实现算法')
    plt.xlabel('算法')
    plt.ylabel('执行时间 (秒)')
    plt.title('各算法执行时间对比')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

