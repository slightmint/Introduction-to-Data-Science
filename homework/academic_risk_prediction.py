import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置 LOKY_MAX_CPU_COUNT 环境变量
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 设置为您想使用的核心数

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 1. 数据加载
def load_data():
    # 假设训练集和测试集已经准备好
    train_data = pd.read_csv('train_data.csv', encoding='utf-8')
    test_data = pd.read_csv('test_data.csv', encoding='utf-8')
    return train_data, test_data

# 2. 数据探索
def explore_data(df):
    print(f"数据集维度: {df.shape}")
    print("\n数据类型概览:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    print("\n基本描述性统计:")
    print(df.describe())
    
    # 目标变量分布
    if '学业状态' in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='学业状态', data=df)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom')
        plt.title('学业状态分布')
        plt.tight_layout()
        plt.savefig('学业状态分布.png')
        plt.close()
        
        # 计算各类别比例
        status_counts = df['学业状态'].value_counts(normalize=True) * 100
        print("\n学业状态分布比例:")
        print(status_counts)

# 3. 特征工程
def feature_engineering(train_df, test_df):
    # 合并数据集以进行一致的特征工程
    test_has_target = '学业状态' in test_df.columns
    
    if test_has_target:
        all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    else:
        # 为测试集创建占位符目标列
        test_df_temp = test_df.copy()
        test_df_temp['学业状态'] = '未知'
        all_data = pd.concat([train_df, test_df_temp], axis=0, ignore_index=True)
    
    # 标记训练集和测试集
    all_data['is_train'] = 0
    all_data.loc[:len(train_df)-1, 'is_train'] = 1
    
    # 基础特征提取和转换
    
    # 1. 计算学期表现差异
    for metric in ['获学分课程数', '选课数', '参加考试数', '通过课程数', '平均成绩', '未参加考试课程数']:
        all_data[f'{metric}_差异'] = all_data[f'第2学期{metric}'] - all_data[f'第1学期{metric}']
    
    # 2. 计算比率特征
    # 第一学期
    all_data['第1学期通过率'] = all_data['第1学期通过课程数'] / all_data['第1学期选课数'].replace(0, 1)
    all_data['第1学期参考率'] = all_data['第1学期参加考试数'] / all_data['第1学期选课数'].replace(0, 1)
    all_data['第1学期获学分比例'] = all_data['第1学期获学分课程数'] / all_data['第1学期选课数'].replace(0, 1)
    
    # 第二学期
    all_data['第2学期通过率'] = all_data['第2学期通过课程数'] / all_data['第2学期选课数'].replace(0, 1)
    all_data['第2学期参考率'] = all_data['第2学期参加考试数'] / all_data['第2学期选课数'].replace(0, 1)
    all_data['第2学期获学分比例'] = all_data['第2学期获学分课程数'] / all_data['第2学期选课数'].replace(0, 1)
    
    # 整体表现
    all_data['总选课数'] = all_data['第1学期选课数'] + all_data['第2学期选课数']
    all_data['总通过课程数'] = all_data['第1学期通过课程数'] + all_data['第2学期通过课程数']
    all_data['总通过率'] = all_data['总通过课程数'] / all_data['总选课数'].replace(0, 1)
    all_data['平均成绩'] = (all_data['第1学期平均成绩'] + all_data['第2学期平均成绩']) / 2
    
    # 3. 家庭背景特征
    all_data['父母学历差异'] = all_data['父亲学历'] - all_data['母亲学历']
    all_data['父母职业类别相同'] = (all_data['父亲职业'] == all_data['母亲职业']).astype(int)
    
    # 4. 经济因素交互
    all_data['经济压力指数'] = all_data['是否欠费'] + (1 - all_data['学费是否按时缴纳']) + (1 - all_data['是否奖学金'])
    all_data['经济因素*GDP'] = all_data['经济压力指数'] * all_data['GDP增长率']
    all_data['经济因素*失业率'] = all_data['经济压力指数'] * all_data['失业率']
    
    # 5. 入学背景特征
    all_data['志愿优先度'] = 1 / all_data['志愿顺序'].replace(0, 1)
    
    # 分离处理后的训练集和测试集
    train_processed = all_data[all_data['is_train'] == 1].drop('is_train', axis=1)
    test_processed = all_data[all_data['is_train'] == 0].drop('is_train', axis=1)
    
    if not test_has_target:
        test_processed = test_processed.drop('学业状态', axis=1)
    
    return train_processed, test_processed

# 4. 数据预处理
def preprocess_data(train_df, test_df):
    # 分离特征和目标变量
    X_train = train_df.drop(['学生编号', '学业状态'], axis=1)
    
    # 将目标变量转换为数值类型
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['学业状态'])
    
    if '学业状态' in test_df.columns:
        X_test = test_df.drop(['学生编号', '学业状态'], axis=1)
        y_test = label_encoder.transform(test_df['学业状态'])
    else:
        X_test = test_df.drop(['学生编号'], axis=1)
        y_test = None
    
    # 保存标签编码映射，用于后续解码
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"标签编码映射: {label_mapping}")
    
    # 识别数值特征和分类特征
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 应用预处理
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, y_train, X_test_processed, y_test, preprocessor, label_encoder

# 5. 模型训练和评估
def train_and_evaluate_models(X_train, y_train, X_test=None, y_test=None):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        
        # 使用交叉验证评估模型
        cv_accuracy = []
        cv_micro_f1 = []
        cv_macro_f1 = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
            # 修改这里，直接使用索引而不是 iloc
            y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_val_cv)
            
            cv_accuracy.append(accuracy_score(y_val_cv, y_pred))
            cv_micro_f1.append(f1_score(y_val_cv, y_pred, average='micro'))
            cv_macro_f1.append(f1_score(y_val_cv, y_pred, average='macro'))
        
        # 计算平均性能指标
        avg_accuracy = np.mean(cv_accuracy)
        avg_micro_f1 = np.mean(cv_micro_f1)
        avg_macro_f1 = np.mean(cv_macro_f1)
        avg_score = (avg_accuracy + avg_micro_f1 + avg_macro_f1) / 3
        
        print(f"{name} 平均准确率: {avg_accuracy:.4f}")
        print(f"{name} 平均Micro-F1: {avg_micro_f1:.4f}")
        print(f"{name} 平均Macro-F1: {avg_macro_f1:.4f}")
        print(f"{name} 平均综合得分: {avg_score:.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': avg_accuracy,
            'micro_f1': avg_micro_f1,
            'macro_f1': avg_macro_f1,
            'avg_score': avg_score
        }
        
        # 更新最佳模型
        if avg_score > best_score:
            best_score = avg_score
            best_model = name
    
    print(f"\n最佳模型是: {best_model}, 平均综合得分: {results[best_model]['avg_score']:.4f}")
    
    # 在整个训练集上训练最佳模型
    final_model = models[best_model]
    final_model.fit(X_train, y_train)
    
    # 如果有测试集，评估最终模型
    if X_test is not None and y_test is not None:
        y_test_pred = final_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_micro_f1 = f1_score(y_test, y_test_pred, average='micro')
        test_macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        
        print("\n最终测试集评估:")
        print(f"准确率: {test_accuracy:.4f}")
        print(f"Micro-F1: {test_micro_f1:.4f}")
        print(f"Macro-F1: {test_macro_f1:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_test_pred))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_train), 
                    yticklabels=np.unique(y_train))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig('混淆矩阵.png')
        plt.close()
    
    return final_model, results

# 6. 特征重要性分析
def feature_importance_analysis(model, feature_names):
    try:
        # 尝试获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            print("此模型不支持直接提取特征重要性")
            return
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        indices = np.argsort(importances)[-20:]  # 显示前20个重要特征
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title('前20个重要特征')
        plt.tight_layout()
        plt.savefig('特征重要性.png')
        plt.close()
        
        # 返回特征重要性数据
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '重要性': importances
        })
        return importance_df.sort_values('重要性', ascending=False)
    
    except Exception as e:
        print(f"特征重要性分析出错: {e}")
        return None

# 7. 超参数调优
def hyperparameter_tuning(X_train, y_train, best_model_name):
    print("\n开始超参数调优...")
    
    if best_model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
    elif best_model_name == 'XGBoost':
        model = xgb.XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif best_model_name == 'LightGBM':
        model = lgb.LGBMClassifier(random_state=42, verbose=-1)  # 添加 verbose=-1 减少输出
        param_grid = {
            'n_estimators': [100, 300],  # 减少选项
            'learning_rate': [0.01, 0.2],  # 减少选项
            'num_leaves': [31, 100],  # 减少选项
            'max_depth': [-1, 20]  # 减少选项
        }
    else:  # LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2']
        }
    
    # 执行网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 8. 生成预测结果
def generate_predictions(model, X_test, test_ids, label_encoder=None):
    predictions = model.predict(X_test)
    
    # 如果有标签编码器，将数值预测转换回原始标签
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    # 创建提交文件
    submission = pd.DataFrame({
        '学生编号': test_ids,
        '学业状态': predictions
    })
    
    # 保存结果
    submission.to_csv('姓名-题目1预测结果.csv', index=False, encoding='utf-8')
    print("预测结果已保存至 '姓名-题目1预测结果.csv'")
    
    return submission

# 主函数
def main():
    print("加载数据...")
    train_data, test_data = load_data()
    
    print("\n探索训练数据...")
    explore_data(train_data)
    
    print("\n执行特征工程...")
    train_processed, test_processed = feature_engineering(train_data, test_data)
    
    print("\n预处理数据...")
    X_train, y_train, X_test, y_test, preprocessor, label_encoder = preprocess_data(train_processed, test_processed)
    
    print("\n训练和评估多个模型...")
    best_model, results = train_and_evaluate_models(X_train, y_train, X_test, y_test)
    
    # 确定最佳模型名称
    best_model_name = max(results, key=lambda k: results[k]['avg_score'])
    
    print("\n分析特征重要性...")
    # 获取特征名称（处理OneHotEncoder的情况）
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # 获取OneHotEncoder的特征名称
            for i, feature in enumerate(features):
                categories = transformer.named_steps['onehot'].categories_[i]
                for category in categories:
                    feature_names.append(f"{feature}_{category}")
        else:
            feature_names.extend(features)
    
    importance_df = feature_importance_analysis(best_model, feature_names)
    if importance_df is not None:
        print("\n前10个重要特征:")
        print(importance_df.head(10))
    
    print("\n对最佳模型进行超参数调优...")
    tuned_model = hyperparameter_tuning(X_train, y_train, best_model_name)
    
    print("\n生成最终预测...")
    test_ids = test_data['学生编号']
    submission = generate_predictions(tuned_model, X_test, test_ids, label_encoder)
    
    print("\n任务完成!")

if __name__ == "__main__":
    main()