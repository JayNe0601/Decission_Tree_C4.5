import numpy as np
import pandas as pd
import C4_5
from sklearn.model_selection import train_test_split

# 读取数据集并预处理
def ReadDataSet(path):
    '''
    | 特征名称 | 备注 | 取值 |
    | --- | --- | --- | --- |
    # | ID | 编号，从1开始 | 正整数 |
    | Survived | 表示生还与否，1代表生还、0代表遇难 | 0、1 |
    | Class | 机票类型 | 0、1、2 |
    # | Name | 姓名，可以是中文，也可以是英文 | 字符串 |
    | Sex | 性别 | male、female |
    | Age | 年龄 | 浮点数 |
    # | Siblings | 登机中兄弟姐妹个数 | 整数 |
    # | Parents_and_children | 登机中父母与子女个数 | 整数 |
    # | Air_ticket_ID | 机票编号 | 字符串 |
    # | Air_ticket_fare | 机票价格 | 浮点数 |
    # | Seat_number | 座位号 | 字符串 |
    | City | 登机城市，只有C、S、Q三个地方登机 | C、S、Q |
    '''
    # 读取数据集
    data = pd.read_csv(path)
    # 查看数据集信息和前五行的数据内容
    print('初始数据集信息：')
    print(data.info())
    print('初始数据集前五行：')
    print(data[:5])
    '''
    删去ID（编号）、Name（姓名）、Air_ticket_ID（机票编号）、Siblings（登机中兄弟姐妹个数）、
    'Parents_and_children'（孩子父母数量）、Air_ticket_fare（机票价格）、Seat_number（座位号）
    编号和机票编号和座位号是唯一的，对模型建立没有意义
    机票类型决定机票价格，无需再加入机票价格
    姓名对于模型训练没有意义
    兄弟姐妹数量和父母数量对模型的训练意义不大
    '''
    data.drop(columns = ['ID', 'Name', 'Air_ticket_ID', 'Parents_and_children', 'Seat_number', 'Siblings', 'Air_ticket_fare'], inplace = True)
    print('删去后的数据集信息：')
    print(data.info())

    # 针对非离散值，根据数据范围划分出10个分类点，对于有限个取值的离散值，将其转化为整数
    bin = 10 # 划分的分类数
    feat_ranges = {} # 用与存储分类点
    cont_feat = ['Age'] # 连续值特征
    for feat in cont_feat:
        # 计算分类点
        min_val = np.nanmin(data[feat])
        max_val = np.nanmax(data[feat])
        feat_ranges[feat] = np.linspace(min_val, max_val, bin).tolist()
        # 查看分类点
        print(feat, ':')
        for spt in feat_ranges[feat]:
            print(spt)

    cat_feat = ['Sex', 'Class', 'City'] # 离散值特征
    for feat in cat_feat:
        data[feat] = data[feat].astype('category') # 将离散值转化为分类数据
        print(feat, ':', data[feat].cat.categories) # 查看分类
        data[feat] = data[feat].cat.codes.tolist()  # 将分类数据按顺序转化为整数
        ranges = list(set(data[feat]))
        ranges.sort()
        feat_ranges[feat] = ranges
    # 针对缺失值
    data.fillna(-1, inplace = True) # 将缺失值填充为-1
    # 遍历查看分类点
    for feat in feat_ranges.keys():
        feat_ranges[feat] = [-1] + feat_ranges[feat] # 将缺失值也作为一个分类点
        print(feat, ':', feat_ranges[feat])

    # 划分数据集
    np.random.seed(0) # 设置随机种子, 为了保证每次划分数据集的结果一致
    feat_names = data.columns[1: ]
    label_name = data.columns[0]
    X = np.array(data[feat_names])
    y = np.array(data[label_name])
    # 训练集和测试集划分 7 : 3
    print(X)
    print(y)
    # 训练集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    print('训练集大小：', len(X_train))
    print('测试集大小：', len(X_test))
    print('特征数：', X_train.shape[1])
    return X_train, X_test, y_train, y_test, feat_ranges, feat_names

if __name__ == '__main__':
    # 读取数据集的路径
    path = r'E:\Develop\Decission_Tree_C4.5\Deission_Tree_C4.5\data_binary.csv'
    # 数据预处理
    X_train, X_test, y_train, y_test, feat_ranges, feat_names = ReadDataSet(path)
    # 创建C4.5决策树对象
    DT = C4_5.DeissionTree(X_train, y_train, feat_ranges, feat_names)
    print('叶子节点数量：', DT.T)
    # 测试集与训练集的准确率
    print('训练集准确率', DT.accuracy(X_train, y_train))
    print('测试集准确率', DT.accuracy(X_test, y_test))
    # 绘制测试集ROC曲线
    DT.ROC(X_test, y_test)
    # 绘制训练集ROC曲线
    DT.ROC(X_train, y_train)