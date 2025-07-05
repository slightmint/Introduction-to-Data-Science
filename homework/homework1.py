import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = "house.csv"  # 如果在本地运行请确保文件名是 house.csv

df = pd.read_csv(file_path)

# (1) 缺失值处理
df['green_rate'].fillna(df['green_rate'].mean(), inplace=True)
df['crime_rate'].fillna(df['crime_rate'].mean(), inplace=True)

# (2) 异常值检测 (IQR)
def detect_outliers_IQR(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return len(outliers), round((len(outliers)/len(df))*100, 2)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('id')
outlier_summary = pd.DataFrame(columns=["Outliers", "Outlier Ratio (%)"])
for col in numeric_cols:
    count, ratio = detect_outliers_IQR(col)
    outlier_summary.loc[col] = [count, ratio]

print("\n(2) Outlier Summary:\n", outlier_summary.sort_values(by="Outlier Ratio (%)", ascending=False))

# (3) 相关性分析
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.show()

# (4) price 标准化 (Z-score)
scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['price']])

# (5) price 离散化
# 分三级：Low, Medium, High
df['price_category'] = pd.qcut(df['price'], q=3, labels=['Low', 'Medium', 'High'])

# (6) 与 price 最关联的三个特征
price_corr = corr_matrix['price'].abs().sort_values(ascending=False)
top_3 = price_corr[1:4]  # 第一个是 price 自身
print("\n(6) Top 3 features most correlated with price:\n", top_3)

# 可选：显示 price 与最关联特征的散点图
for feature in top_3.index:
    sns.scatterplot(x=df[feature], y=df['price'])
    plt.title(f"Price vs {feature}")
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.show()
