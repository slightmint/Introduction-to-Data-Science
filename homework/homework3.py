import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
from matplotlib import font_manager

# 忽略警告
warnings.filterwarnings('ignore')

# 配置中文字体支持
# 方法1：使用系统中已有的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 如果上面的方法不起作用，可以尝试方法2：
# 查找系统中的中文字体并使用
# 查找系统中文字体的路径，Windows系统一般在C:\Windows\Fonts
# font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = 'SimHei'

# 自定义一元线性回归类
class SimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """使用最小二乘法拟合一元线性回归模型"""
        # 确保X是一维数组
        X = X.flatten() if X.ndim > 1 else X
        
        # 计算均值
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # 计算斜率
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        
        return self
    
    def predict(self, X):
        """使用拟合的模型进行预测"""
        X = X.flatten() if X.ndim > 1 else X
        return self.w * X + self.b

# 多元线性回归类
class MultipleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """使用正规方程法拟合多元线性回归模型"""
        # 添加偏置项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 计算参数 
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.b = theta[0]
        self.w = theta[1:]
        
        return self
    
    def predict(self, X):
        """使用拟合的模型进行预测"""
        return X.dot(self.w) + self.b

# 逻辑回归类
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def sigmoid(self, z):
        """sigmoid函数"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """使用梯度下降法拟合逻辑回归模型"""
        # 初始化参数
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            # 计算模型输出
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self.sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 更新参数
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        linear_model = np.dot(X, self.w) + self.b
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# 加载波士顿房价数据集
def load_boston_dataset():
    """加载波士顿房价数据集"""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    try:
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        # 特征名称
        feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
        
        return data, target, feature_names
    except:
        # 如果无法从原始源获取，使用备用方法
        print("无法从原始源获取波士顿数据集，使用备用数据源...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        X = df.drop('medv', axis=1).values
        y = df['medv'].values
        feature_names = df.drop('medv', axis=1).columns.tolist()
        return X, y, feature_names

# 一元线性回归实验
def simple_linear_regression_experiment():
    print("\n===== 一元线性回归实验 =====")
    
    # 加载波士顿房价数据集
    X, y, feature_names = load_boston_dataset()
    
    # 使用房间数量(RM)作为特征
    rm_index = feature_names.index('RM')
    X_rm = X[:, rm_index].reshape(-1, 1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_rm, y, test_size=0.2, random_state=42)
    
    # 使用自定义一元线性回归
    print("使用自定义一元线性回归模型:")
    slr = SimpleLinearRegression()
    slr.fit(X_train, y_train)
    y_pred_custom = slr.predict(X_test)
    
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)
    
    print(f"斜率 (w): {slr.w:.4f}")
    print(f"截距 (b): {slr.b:.4f}")
    print(f"均方误差 (MSE): {mse_custom:.4f}")
    print(f"决定系数 (R²): {r2_custom:.4f}")
    
    # 使用sklearn的线性回归
    print("\n使用sklearn的线性回归模型:")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)
    
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"斜率 (w): {lr.coef_[0]:.4f}")
    print(f"截距 (b): {lr.intercept_:.4f}")
    print(f"均方误差 (MSE): {mse_sklearn:.4f}")
    print(f"决定系数 (R²): {r2_sklearn:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='实际值')
    plt.plot(X_test, y_pred_custom, color='red', linewidth=2, label='自定义模型预测')
    plt.plot(X_test, y_pred_sklearn, color='green', linewidth=2, label='sklearn模型预测')
    plt.xlabel('平均房间数 (RM)')
    plt.ylabel('房价 (千美元)')
    plt.title('一元线性回归: 房间数量与房价关系')
    plt.legend()
    plt.savefig('simple_linear_regression.png')
    plt.close()

# 多元线性回归实验
def multiple_linear_regression_experiment():
    print("\n===== 多元线性回归实验 =====")
    
    # 加载波士顿房价数据集
    X, y, feature_names = load_boston_dataset()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用自定义多元线性回归
    print("使用自定义多元线性回归模型:")
    mlr = MultipleLinearRegression()
    mlr.fit(X_train, y_train)
    y_pred_custom = mlr.predict(X_test)
    
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)
    
    print(f"截距 (b): {mlr.b:.4f}")
    print(f"均方误差 (MSE): {mse_custom:.4f}")
    print(f"决定系数 (R²): {r2_custom:.4f}")
    
    # 打印特征重要性
    feature_importance = list(zip(feature_names, mlr.w))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n特征重要性 (按绝对值排序):")
    for feature, weight in feature_importance:
        print(f"{feature}: {weight:.4f}")
    
    # 使用sklearn的线性回归
    print("\n使用sklearn的线性回归模型:")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_sklearn = lr.predict(X_test)
    
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    
    print(f"截距 (b): {lr.intercept_:.4f}")
    print(f"均方误差 (MSE): {mse_sklearn:.4f}")
    print(f"决定系数 (R²): {r2_sklearn:.4f}")
    
    # 可视化预测结果对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_custom, color='red', label='自定义模型预测')
    plt.scatter(y_test, y_pred_sklearn, color='green', alpha=0.5, label='sklearn模型预测')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('实际房价')
    plt.ylabel('预测房价')
    plt.title('多元线性回归: 预测值与实际值对比')
    plt.legend()
    plt.savefig('multiple_linear_regression.png')
    plt.close()

# 逻辑回归实验
def logistic_regression_experiment():
    print("\n===== 逻辑回归实验 =====")
    
    # 加载乳腺癌数据集
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 特征标准化
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std
    
    # 使用自定义逻辑回归
    print("使用自定义逻辑回归模型:")
    lr_custom = LogisticRegressionCustom(learning_rate=0.01, n_iterations=1000)
    lr_custom.fit(X_train_scaled, y_train)
    y_pred_custom = lr_custom.predict(X_test_scaled)
    
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    
    print(f"准确率: {accuracy_custom:.4f}")
    print("混淆矩阵:")
    print(cm_custom)
    
    # 使用sklearn的逻辑回归
    print("\n使用sklearn的逻辑回归模型:")
    lr_sklearn = LogisticRegression(max_iter=1000)
    lr_sklearn.fit(X_train_scaled, y_train)
    y_pred_sklearn = lr_sklearn.predict(X_test_scaled)
    
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    
    print(f"准确率: {accuracy_sklearn:.4f}")
    print("混淆矩阵:")
    print(cm_sklearn)
    
    # 可视化ROC曲线
    from sklearn.metrics import roc_curve, auc
    
    # 计算预测概率
    y_prob_custom = lr_custom.predict_proba(X_test_scaled)
    y_prob_sklearn = lr_sklearn.predict_proba(X_test_scaled)[:, 1]
    
    # 计算ROC曲线
    fpr_custom, tpr_custom, _ = roc_curve(y_test, y_prob_custom)
    fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, y_prob_sklearn)
    
    # 计算AUC
    roc_auc_custom = auc(fpr_custom, tpr_custom)
    roc_auc_sklearn = auc(fpr_sklearn, tpr_sklearn)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_custom, tpr_custom, color='red', lw=2, label=f'自定义模型 (AUC = {roc_auc_custom:.4f})')
    plt.plot(fpr_sklearn, tpr_sklearn, color='green', lw=2, label=f'sklearn模型 (AUC = {roc_auc_sklearn:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig('logistic_regression_roc.png')
    plt.close()

# 主函数
def main():
    print("回归算法实验")
    
    # 一元线性回归实验
    simple_linear_regression_experiment()
    
    # 多元线性回归实验
    multiple_linear_regression_experiment()
    
    # 逻辑回归实验
    logistic_regression_experiment()

if __name__ == "__main__":
    main()