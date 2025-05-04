import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 定义树的节点
class Node:
    # 初始化
    def __init(self):
        # 内部节点的feat表示用来分类的特征编号，数字与数据中的顺序对应
        # 叶节点的feat表示对应的分类结果
        self.feat = None
        # 分类值列表，表示按照其中的值向子节点分类
        self.split = None
        # 子节点列表，叶节点的子节点列表为空
        self.child = []

# 定义决策树
class DeissionTree:
    # 初始化
    def __init__(self, X, y, feat_ranges, feat_names):
        self.root = Node()
        self.X = X
        self.y = y
        self.feat_ranges = feat_ranges # 特征取值范围
        self.feat_names = feat_names # 特征名称
        self.eps = 1e-8 # 防止log(0)和除0错误
        self.T = 0 # 叶节点的分类个数
        self.C4_5(self.root, self.X, self.y) # 构造决策树
        self.visualize_tree() # 可视化决策树

    # 计算a * log(a)
    def aloga(self, a):
        return a * np.log2(a + self.eps)

    # 计算某个子数据集的熵
    def entropy(self, y):
        # 总数
        N = len(y)
        if N == 0:
            return 0
        # 统计每个类别出现的次数
        cnt = np.unique(y, return_counts=True) # 统计每个类别出现的次数
        # 计算熵
        Ent = 0 - np.sum([self.aloga(t / N) for t in cnt])
        return Ent

    # 计算特征划分 feat <= val 划分数据集的信息增益
    def info_gain(self, X, y, feat, val):
        # 计算划分前的熵
        N = len(y)
        if N == 0:
            return 0
        Ent_pre = self.entropy(y)
        # 计算划分后的熵
        Ent_now = 0
        # 分别计算划分后的熵
        y_l = y[X[:, feat] <= val]
        Ent_now += (len(y_l) / N * self.entropy(y_l))
        y_r = y[X[:, feat] > val]
        Ent_now += (len(y_r) / N * self.entropy(y_r))
        Gain = Ent_pre - Ent_now
        return Gain

    # 计算特征划分 feat <= val 划分数据集的属性的固有值
    def intrinsic_value(self, X, y, feat, val):
        N = len(y)
        if N == 0:
            return 0
        IV = 0
        y_l = y[X[:, feat] <= val]
        IV += -self.aloga(len(y_l) / N)
        y_r = y[X[:, feat] > val]
        IV += -self.aloga(len(y_r) / N)
        return IV

    # 计算特征划分 feat <= val 划分数据集的信息增益率
    def info_gain_ratio(self, X, y, feat, val):
        # 计算信息增益
        Gain = self.info_gain(X, y, feat, val)
        # 计算固有值
        IV = self.intrinsic_value(X, y, feat, val)
        return Gain / IV

    # 递归分裂节点，构造决策树
    def C4_5(self, node, X, y):
        # 如果y中样本全为同一类，则以这一类作为叶子节点
        if len(np.unique(y)) == 1:
            node.feat = y[0]
            self.T += 1
            node.split = None
            node.child = []
            print('y中样本全为同一类，叶子节点，分类为：', node.feat)
            return
        # 如果X中样本全为同一类或者属性为空，则以这一类中最多的作为叶子节点
        elif len(np.unique(X)) == 1 or len(X) == 0:
            vals, cnt = np.unique(y, return_counts=True)
            node.feat = vals[np.argmax(cnt)]
            self.T += 1
            node.split = None
            node.child = []
            print('X中样本全为同一类或者属性为空，叶子节点，分类为：', node.feat)
            return

        # 寻找最优分类特征和分类点
        Gs = [] # 存取信息增益
        GRs = [] # 存取信息增益率
        feats = [] # 存取特征
        vals = [] # 存取分类点
        best_GR = 0 # 最优信息增益率
        best_feat = None # 最优特征
        best_val = None # 最优分类点
        # 遍历特征
        for feat in range(len(self.feat_names)):
            # 遍历特征的取值范围
            for val in self.feat_ranges[self.feat_names[feat]]:
                # 计算信息增益率
                GR = self.info_gain_ratio(X, y, feat, val)
                # 计算信息增益
                G = self.info_gain(X, y, feat, val)
                Gs.append(G)
                GRs.append(GR)
                feats.append(feat)
                vals.append(val)

        # 寻找最优信息增益率（从候选集中划分信息增益高于平均水平的属性与划分，再从中找出增益率最高的）
        avg_G = np.mean(Gs)
        for i in range(len(GRs)):
            if Gs[i] > avg_G:
                if GRs[i] > best_GR:
                    best_GR = GRs[i]
                    best_feat = feats[i]
                    best_val = vals[i]
        # 输出最优特征和分类点
        print(f'best_feat: {best_feat}, best_val: {best_val}, best_GR: {best_GR}')
        # 如果存在最优，执行分裂
        if best_feat != None:
            X_feat = X[:, best_feat]
            node.feat = best_feat
            node.split = best_val
            l_child = Node()
            l_X = X[X_feat <= best_val]
            l_y = y[X_feat <= best_val]
            # 递归分裂左子节点
            self.C4_5(l_child, l_X, l_y)
            r_child = Node()
            r_X = X[X_feat > best_val]
            r_y = y[X_feat > best_val]
            # 递归分裂右子节点
            self.C4_5(r_child, r_X, r_y)
            node.child = [l_child, r_child]

        # 否则，返回y中最多的作为叶子节点
        else:
            vals, cnt = np.unique(y, return_counts=True)
            node.feat = vals[np.argmax(cnt)]
            self.T += 1
            node.split = None
            node.child = []
            print('叶子节点，分类为：', node.feat)

    # 预测
    def predict(self, X):
        node = self.root
        while node.split != None:
            if X[node.feat] <= node.split:
                node = node.child[0]
            else:
                node = node.child[1]
        return node.feat

    # 计算准确率
    def accuracy(self, X, y):
        currect = 0
        for X_i, y_i in zip(X, y):
            pred = self.predict(X_i)
            if pred == y_i:
                currect += 1
        return currect / len(y)

    # 可视化决策树
    def plot_tree(self, node, x, y, dx, dy, ax, boolean = 0):
        # 绘制当前节点
        if node.split is None:
            s = 'No' if node.feat == 0 else 'Yes'
            ax.text(x, y, s, ha='center', va='center', fontsize=6,
                    bbox=dict(boxstyle='round', facecolor='lightblue'))
        else:
            ax.text(x, y, f'{self.feat_names[node.feat]} <= {round(node.split, 4)}', ha='center', va='center',
                    fontsize=6, bbox=dict(boxstyle='round', facecolor='lightgreen'))
            ax.plot([x, x - dx], [y, y - dy], 'k-')
            self.plot_tree(node.child[0], x - dx, y - dy, dx / 2, dy, ax, 1)
            # 绘制右子树
            ax.plot([x, x + dx], [y, y - dy], 'k-')
            self.plot_tree(node.child[1], x + dx, y - dy, dx / 2, dy, ax, 2)

    # 在 DeissionTree 类中调用
    def visualize_tree(self):
        fig, ax = plt.subplots(figsize=(20, 12))  # 调整界面大小
        ax.axis('off')
        self.plot_tree(self.root, 0, 0, 0.5, 0.1, ax, 0)  # 调整节点间距
        plt.show()

    # 绘制 ROC 曲线
    def ROC(self, X, y):
        y_pred = [self.predict(x) for x in X]
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == '__main__':
    print("C4.5决策树")