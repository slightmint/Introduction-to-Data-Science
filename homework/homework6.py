# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import time
import matplotlib as mpl

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 数据集基本信息
print("数据集信息:")
print(f"样本数量: {X.shape[0]}")
print(f"特征数量: {X.shape[1]}")
print(f"类别分布: {np.bincount(y)}")
print(f"特征名称: {data.feature_names}")

# 数据预处理：划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义评估模型的函数
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 记录训练结束时间
    train_time = time.time() - start_time
    
    # 在训练集和测试集上进行预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # 打印结果
    print(f"\n{model_name} 评估结果:")
    print(f"训练时间: {train_time:.4f}秒")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_test_pred))
    
    # 返回测试集准确率和训练时间
    return test_accuracy, train_time

# 1. 基础模型 - 决策树
print("\n" + "="*50)
print("基础模型 - 决策树")
base_dt = DecisionTreeClassifier(random_state=42)
dt_accuracy, dt_time = evaluate_model(base_dt, X_train_scaled, X_test_scaled, y_train, y_test, "决策树")

# 2. 基础模型 - SVM
print("\n" + "="*50)
print("基础模型 - SVM")
base_svm = SVC(random_state=42)
svm_accuracy, svm_time = evaluate_model(base_svm, X_train_scaled, X_test_scaled, y_train, y_test, "SVM")

# 3. AdaBoost 集成
print("\n" + "="*50)
print("AdaBoost 集成")
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_accuracy, ada_time = evaluate_model(ada_clf, X_train_scaled, X_test_scaled, y_train, y_test, "AdaBoost")

# 4. Bagging 集成
print("\n" + "="*50)
print("Bagging 集成")
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    random_state=42
)
bag_accuracy, bag_time = evaluate_model(bag_clf, X_train_scaled, X_test_scaled, y_train, y_test, "Bagging")

# 5. 随机森林集成
print("\n" + "="*50)
print("随机森林集成")
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
rf_accuracy, rf_time = evaluate_model(rf_clf, X_train_scaled, X_test_scaled, y_train, y_test, "随机森林")

# 参数调优
print("\n" + "="*50)
print("参数调优")

# AdaBoost 参数调优
print("\nAdaBoost 参数调优:")
ada_param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]  # 修改这里：base_estimator 改为 estimator
}

ada_grid = GridSearchCV(
    AdaBoostClassifier(
        DecisionTreeClassifier(random_state=42),
        random_state=42
    ),
    ada_param_grid,
    cv=5,
    scoring='accuracy'
)

ada_grid.fit(X_train_scaled, y_train)
print(f"最佳参数: {ada_grid.best_params_}")
print(f"最佳交叉验证得分: {ada_grid.best_score_:.4f}")

# 使用最佳参数的AdaBoost
best_ada = ada_grid.best_estimator_
best_ada_accuracy, best_ada_time = evaluate_model(best_ada, X_train_scaled, X_test_scaled, y_train, y_test, "调优后的AdaBoost")

# Bagging 参数调优
print("\nBagging 参数调优:")
bag_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0]
}

bag_grid = GridSearchCV(
    BaggingClassifier(
        DecisionTreeClassifier(random_state=42),
        bootstrap=True,
        random_state=42
    ),
    bag_param_grid,
    cv=5,
    scoring='accuracy'
)

bag_grid.fit(X_train_scaled, y_train)
print(f"最佳参数: {bag_grid.best_params_}")
print(f"最佳交叉验证得分: {bag_grid.best_score_:.4f}")

# 使用最佳参数的Bagging
best_bag = bag_grid.best_estimator_
best_bag_accuracy, best_bag_time = evaluate_model(best_bag, X_train_scaled, X_test_scaled, y_train, y_test, "调优后的Bagging")

# 随机森林参数调优
print("\n随机森林参数调优:")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy'
)

rf_grid.fit(X_train_scaled, y_train)
print(f"最佳参数: {rf_grid.best_params_}")
print(f"最佳交叉验证得分: {rf_grid.best_score_:.4f}")

# 使用最佳参数的随机森林
best_rf = rf_grid.best_estimator_
best_rf_accuracy, best_rf_time = evaluate_model(best_rf, X_train_scaled, X_test_scaled, y_train, y_test, "调优后的随机森林")

# 结果可视化
print("\n" + "="*50)
print("模型性能比较")

# 准确率比较
models = ['决策树', 'SVM', 'AdaBoost', 'AdaBoost(调优)', 'Bagging', 'Bagging(调优)', '随机森林', '随机森林(调优)']
accuracies = [dt_accuracy, svm_accuracy, ada_accuracy, best_ada_accuracy, 
              bag_accuracy, best_bag_accuracy, rf_accuracy, best_rf_accuracy]
times = [dt_time, svm_time, ada_time, best_ada_time, 
         bag_time, best_bag_time, rf_time, best_rf_time]

# 绘制准确率对比图
plt.figure(figsize=(12, 6))
plt.bar(models, accuracies, color=['blue', 'blue', 'green', 'green', 'orange', 'orange', 'red', 'red'])
plt.xlabel('模型')
plt.ylabel('测试集准确率')
plt.title('不同模型的测试集准确率对比')
plt.ylim(0.9, 1.0)  # 调整Y轴范围以便更好地显示差异
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')

# 绘制训练时间对比图
plt.figure(figsize=(12, 6))
plt.bar(models, times, color=['blue', 'blue', 'green', 'green', 'orange', 'orange', 'red', 'red'])
plt.xlabel('模型')
plt.ylabel('训练时间(秒)')
plt.title('不同模型的训练时间对比')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_time_comparison.png')

# 结论分析
print("\n" + "="*50)
print("结论分析:")
print("1. 集成方法与单个模型的对比:")
print(f"   - 决策树准确率: {dt_accuracy:.4f}")
print(f"   - SVM准确率: {svm_accuracy:.4f}")
print(f"   - 最佳AdaBoost准确率: {best_ada_accuracy:.4f}")
print(f"   - 最佳Bagging准确率: {best_bag_accuracy:.4f}")
print(f"   - 最佳随机森林准确率: {best_rf_accuracy:.4f}")

print("\n2. 不同集成方法的对比:")
print(f"   - AdaBoost: 基础准确率 {ada_accuracy:.4f}, 调优后 {best_ada_accuracy:.4f}")
print(f"   - Bagging: 基础准确率 {bag_accuracy:.4f}, 调优后 {best_bag_accuracy:.4f}")
print(f"   - 随机森林: 基础准确率 {rf_accuracy:.4f}, 调优后 {best_rf_accuracy:.4f}")

print("\n3. 综合分析:")
best_model = models[accuracies.index(max(accuracies))]
print(f"   - 在本实验中，{best_model}模型表现最佳，准确率达到{max(accuracies):.4f}")
print("   - 集成学习方法普遍优于单个基础模型")
print("   - 参数调优对模型性能有显著提升")
print("   - 随机森林在不调参的情况下也有很好的表现，说明其对乳腺癌数据集有较好的适应性")