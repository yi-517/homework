import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 读取两个csv文件，合并为一个
    column_names = ['country', 'description', 'designation', 'points', 'price',
                    'province', 'region_1', 'region_2', 'variety', 'winery']
    wine_data1 = pd.read_csv("original_data/winemag-data-130k-v2.csv", usecols=column_names)
    wine_data2 = pd.read_csv("original_data/winemag-data_first150k.csv", usecols=column_names)
    wine_data = pd.concat([wine_data2, wine_data1], ignore_index=True, sort=False)
    print(wine_data)

    # 去重,发现无重复数据
    wine_data.duplicated().value_counts()
    print(wine_data)

    # 数据摘要
    # 标称属性，给出每个可能取值的频数
    # country
    group_by_country = wine_data.groupby('country').size().sort_values()
    print(group_by_country)
    # designation
    group_by_designation = wine_data.groupby('designation').size().sort_values()
    print(group_by_designation)
    # province
    group_by_province = wine_data.groupby('province').size().sort_values()
    print(group_by_province)
    # region_1
    group_by_region_1 = wine_data.groupby('region_1').size().sort_values()
    print(group_by_region_1)
    # region_2
    group_by_region_2 = wine_data.groupby('region_2').size().sort_values()
    print(group_by_region_2)
    # variety
    group_by_variety = wine_data.groupby('variety').size().sort_values()
    print(group_by_variety)
    # winery
    group_by_winery = wine_data.groupby('winery').size().sort_values()
    print(group_by_winery)

    # 数值属性，给出5数概况及缺失值的个数
    # 五数概况
    print(wine_data.describe())
    # 缺失值个数
    print(wine_data.isnull().sum())

    # 数据可视化
    # points
    # 直方图
    plt.hist(x=wine_data['points'],
             bins=21,
             edgecolor='black')
    plt.xlabel('points')
    plt.ylabel('number')
    plt.show()
    # 盒图
    plt.boxplot(wine_data['points'])
    plt.show()

    # price
    # 去除缺失值
    df = wine_data['price'].dropna(axis=0, inplace=False)
    # 直方图
    plt.hist(x=df,
             bins=34,
             edgecolor='black')
    plt.xlabel('price')
    plt.ylabel('number')
    plt.show()
    # 盒图
    plt.boxplot(df)
    plt.show()
    # 散点图
    sns.stripplot(y='price', data=wine_data)
    plt.show()

    # 数据缺失值处理
    print(wine_data.isnull().sum())

    # 将缺失部分剔除
    df1 = wine_data.dropna(axis=0, inplace=False)
    print(df1)
    print(df1.isnull().sum())

    # 用最高频率值来填补空缺值
    # price计算频率
    group_by_price = wine_data.groupby('price').size().sort_values()
    print(group_by_price)
    # price 缺失值用20替代
    # country 缺失值用US替代
    # designation 缺失值用Reserve替代
    # province 缺失值用California替代
    # region_1 缺失值用Napa Valley替代
    # region_2 缺失值用Central Coast替代
    # variety 缺失值用Noir替代
    values = {'price': 20, 'country': 'US', 'designation': 'Reserve', 'province': 'California',
              'region_1': 'Napa Valley', 'region_2': 'Central Coast', 'variety': 'Noir'}
    df2 = wine_data.fillna(value=values, inplace=False)
    print(df2.isnull().sum())

    # 通过属性的相关关系来填补缺失值
    # 由于本数据集中只有points和price是数值属性，且只有price有缺失值。
    # 因此我用points和price建立线性回归模型进行price缺失值的填充。
    # 根据上面对price数据的分析，price大多集中在500以下，因此我们用500以下的数据进行建模
    # 含有缺失值的数据组成预测子集
    price_pred = wine_data[np.isnan(wine_data['price'])]
    print(price_pred)
    x_pred = price_pred['points']
    y_pred = price_pred['price']
    # price在0~500的数据组成训练子集
    price_train = wine_data.dropna(subset=['price'], axis=0)
    price_train = price_train[price_train['price'] < 500]
    print(price_train)
    plt.figure()
    plt.xlabel("points")
    plt.ylabel("price")
    plt.grid(True)
    plt.plot(price_train['points'], price_train['price'])
    plt.show()
    # 建模
    line_reg = LinearRegression()
    line_reg.fit(price_train['points'].values.reshape(-1, 1),
                 price_train['price'].values.reshape(-1, 1))
    bias = line_reg.intercept_
    weight = line_reg.coef_
    print("bias: ", bias)
    print("weight: ", weight)
    # 预测缺失值并填补
    y_pred = line_reg.predict(x_pred.values.reshape(-1, 1))
    price_pred['price'] = y_pred
    print(price_pred['price'])
    # 合并还原
    df3 = price_train.append(price_pred)
    print(df3)
    print(df3.isnull().sum())

    # 通过数据对象之间的相似性来填补缺失值
    # 填补缺失值。思路：对数据集进行排序，相邻数据具有高度相似性，用前面一行或后面一行数据的对应值填充缺失值。
    column_names = ['country', 'description', 'designation', 'points',
                    'province', 'region_1', 'region_2',
                    'variety', 'winery']
    df4 = wine_data.sort_values(by=column_names)
    df4.fillna(method='ffill', inplace=True)
    df4.fillna(method='bfill', inplace=True)
    print(df4)
    print(df4.isnull().sum())




