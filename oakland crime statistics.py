import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':

    # 读取csv文件，合并为一个
    column_names = ['Area Id', 'Beat', 'Priority', 'Incident Type Id',
                    'Incident Type Description', 'Event Number']
    records_data1 = pd.read_csv("original_data/records-for-2011.csv",
                                usecols=column_names, dtype='str')
    # print(records_data1.columns)
    records_data2 = pd.read_csv("original_data/records-for-2012.csv",
                                usecols=column_names, dtype='str')
    # print(records_data2.columns)
    records_data3 = pd.read_csv("original_data/records-for-2013.csv",
                                usecols=column_names, dtype='str')
    # print(records_data3.columns)
    records_data4 = pd.read_csv("original_data/records-for-2014.csv",
                                usecols=column_names, dtype='str')
    # print(records_data4.columns)
    records_data5 = pd.read_csv("original_data/records-for-2015.csv",
                                usecols=column_names, dtype='str')
    # print(records_data5.columns)
    records_data6 = pd.read_csv("original_data/records-for-2016.csv",
                                usecols=column_names, dtype='str')
    # print(records_data6.columns)
    records_data = pd.concat([records_data1, records_data2, records_data3,
                              records_data4, records_data5, records_data6],
                             ignore_index=True, sort=False)
    print(records_data)

    # 去重,发现无重复数据
    records_data.duplicated().value_counts()
    print(records_data)

    # 数据摘要
    # 标称属性，给出每个可能取值的频数

    group_by_Area_Id = records_data.groupby('Area Id').size().sort_values()
    print(group_by_Area_Id)
    
    group_by_Beat = records_data.groupby('Beat').size().sort_values()
    print(group_by_Beat)

    group_by_Priority = records_data.groupby('Priority').size().sort_values()
    print(group_by_Priority)
    
    group_by_Incident_Type_Id = records_data.groupby('Incident Type Id').\
        size().sort_values()
    print(group_by_Incident_Type_Id)
    
    group_by_Incident_Type_Description = records_data\
        .groupby('Incident Type Description').size().sort_values()
    print(group_by_Incident_Type_Description)

    group_by_Event_Number = records_data.groupby('Event Number')\
        .size().sort_values()
    print(group_by_Event_Number)

    # 数值属性，给出5数概况及缺失值的个数
    # 五数概况
    # print(records_data.describe())
    # 缺失值个数
    print(records_data.isnull().sum())

    # 数据可视化
    # Area Id 柱状图
    plt.barh(group_by_Area_Id.index, group_by_Area_Id.values)
    plt.title('Area Id 分布')
    plt.show()
    # Beat 柱状图
    plt.figure(figsize=(10, 10))
    plt.barh(group_by_Beat.index, group_by_Beat.values)
    plt.title('Beat 分布')
    plt.show()
    # Priority 柱状图
    plt.figure(figsize=(5, 2))
    plt.barh(group_by_Priority.index, group_by_Priority.values)
    plt.title('Priority 分布')
    plt.show()

    # 数据缺失值处理
    print(records_data.isnull().sum())

    # 将缺失部分剔除
    df1 = records_data.dropna(axis=0, inplace=False)
    print(df1)
    print(df1.isnull().sum())

    # 用最高频率值来填补空缺值
    # Area Id 缺失值用1替代
    # Beat 缺失值用04X替代
    # Priority 缺失值用2替代
    # Incident Type Id 缺失值用933R替代
    # Incident Type Description 缺失值用ALARM-RINGER替代
    values = {'Area Id': '1', 'Beat': '04X', 'Priority': '2',
              'Incident Type Description': 'ALARM-RINGER',
              'Incident Type Id': '933R'}
    df2 = records_data.fillna(value=values, inplace=False)
    print(df2.isnull().sum())

    # 通过属性的相关关系来填补缺失值
    # Incident Type Id和Incident Type Description是高度相关的
    dict_df = records_data[['Incident Type Id', 'Incident Type Description']]\
        .dropna(axis=0, inplace=False)
    dict_df.drop_duplicates(inplace=True)
    dict_df.reset_index()
    print("dict_df")
    print(dict_df)
    print(dict_df.isnull().sum())
    df3 = records_data.drop('Incident Type Description', axis=1)
    print(df3)
    df3 = pd.merge(df3, dict_df)
    print(df3.isnull().sum())

    # 通过数据对象之间的相似性来填补缺失值
    # 思路：对数据集进行排序，相邻数据具有高度相似性，用前面一行数据的对应值填充缺失值。
    column_names = ['Area Id', 'Beat', 'Priority', 'Incident Type Id',
                    'Incident Type Description', 'Event Number']
    df4 = records_data.sort_values(by=column_names)
    df4.fillna(method='ffill', inplace=True)
    print(df4)
    print(df4.isnull().sum())




