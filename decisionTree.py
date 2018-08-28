import numpy as np

class decisionTree:
    """
    使用方法： clf = decisioTree(), 参数mode可选ID3 or C4.5
    - 训练，调用fit方法： clf.fit(X, y) X,y均为np.ndarray类型
    - 预测，调用predict方法：clf.predict(x) X为np.ndarray
    - 可视化决策树结构，调用show()方法
    """
    def __init__(self, mode = 'ID3'):
        self.tree = None

        if mode == 'ID3' or mode == 'C4.5':
            self.mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')

    def calcEntropy(self, y):
        """
        :param y: 数据集的标签
        :return: 熵值
        """
        num = y.shape[0]
        # 统计y中不同laebl值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 0
            labelCounts[label] += 1

        # 计算熵值
        entropy = 0.
        for key in labelCounts:
            prob = 1. * labelCounts[key] / num
            entropy = entropy - prob * np.log2(prob)
        return entropy

    def splitDataSet(self, X, y, index, value):
        """
        :param X:
        :param y:
        :param index:
        :param value:
        :return: 返回数据集中特征下标为index，特征值等于value的子数据集
        """
        ret = []
        featVec = X[:, index]
        X = X[:, [i for i in range(X.shape[1]) if i != index]]

        for i in range(len(featVec)):
            if featVec[i] == value:
                ret.append(i)
        return X[ret, :], y[ret]

    def chooseBestFeatureToSplit_ID3(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        # 特征个数
        numFeatures = X.shape[1]
        # 原始数据集的熵
        oldEntropy = self.calcEntropy(y)
        # 记录最大的信息增益
        bestInfoGain = 0.
        # 信息增益最大时候，所需要选择分割特征的下标
        bestFeatureIndex = -1

        # 对每个特征都计算一下信息增益，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):
            featList = X[:, i]
            uniqueVals = set(featList)
            newEntropy = 0.
            # 对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            # 进一步地可以计算得到第i个特征分割原始数据集后的熵newEntropy

            for value in uniqueVals:
                sub_X, sub_y = self.splitDataSet(X, y, i, value)
                prob = 1. * len(sub_y) / len(y)
                newEntropy = newEntropy + prob * self.calcEntropy(sub_y)

            # 计算信息增益，根据信息增益选择最佳分割特征
            infoGain = oldEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex

    def chooseBestFeatureToSplit_C45(self, X, y):
        numFeatures = X.shape[1]
        oldEntropy = self.calcEntropy(y)
        bestGainRatio = 0.
        bestFeatureIndex = -1
        # 对每个特征都计算一下信息增益比
        for i in range(numFeatures):
            featList = X[:, i]
            uniqueVals = set(featList)
            newEntropy = 0.
            splitInformation = 0.

            for value in uniqueVals:
                sub_X, sub_y = self.splitDataSet(X, y, i, value)
                prob = len(sub_y) * 1. / len(y)
                newEntropy = newEntropy + prob * self.calcEntropy(sub_y)
                splitInformation -= prob * np.log2(prob)

            if splitInformation == 0.0:
                pass
            else:
                infoGain = oldEntropy - newEntropy
                gainRatio = infoGain / splitInformation
                if gainRatio > bestGainRatio:
                    bestGainRatio = gainRatio
                    bestFeatureIndex = i
        return bestFeatureIndex

    def majority(self, labelList):
        """
        :param labelList:
        :return: 返回labelList中出现次数最多的label
        """
        labelCount = {}
        for vote in labelList:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.items(), key=lambda x: x[1], reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, X, y, featureIndex):

        labelList = list(y)
        # 所有label都相同的话，则停止分割，返回该label
        if labelList.count(labelList[0]) == len(labelList):
            return labelList[0]

        # 没有特征可分割时，停止分割，返回出现次数最多的label
        if len(featureIndex) == 0:
            return self.majority(labelList)

        # 可以继续分割的话，选择最佳分割特征
        if self.mode == 'ID3':
            bestFeatIndex = self.chooseBestFeatureToSplit_ID3(X, y)
        else:
            bestFeatIndex = self.chooseBestFeatureToSplit_C45(X, y)

        bestFeatStr = featureIndex[bestFeatIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex  = tuple(featureIndex)
        # 用字典存储决策树。最佳分割特征是key，而对应的键值仍然是一颗树（仍然用字典存储）

        myTree = {bestFeatStr:{}}
        featValues = X[:, bestFeatIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # 对每个value递归的建树
            sub_X, sub_y = self.splitDataSet(X, y, bestFeatIndex, value)
            myTree[bestFeatStr][value] = self.createTree(sub_X, sub_y, featureIndex)
        return myTree

    def fit(self, X, y):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")

        featureIndex = tuple(['x' + str(i) for i in range(X.shape[1])])
        #featureIndex = tuple(['Age', 'Job', 'House', 'Credit'])
        print(featureIndex)
        self.tree = self.createTree(X, y, featureIndex)
        return self

    def predict(self, X):
        if self.tree == None:
            raise NotFittedError("Estimatpr not fitted, call 'fit' first")

        if isinstance(X, np.ndarray):
            pass
        else:
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        def classify(tree, sample):
            featSides = list(tree.keys())
            # featIndex = tree.keys()[0]
            featIndex = featSides[0]
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])]
            valueOfkey = secondDict[key]


            if isinstance(valueOfkey, dict):
                label = classify(valueOfkey, sample)
            else:
                label = valueOfkey
            return label

        if len(X.shape) == 1:
            return classify(self.tree, X)
        else:
            results = []
            for i in range(X.shape[0]):
                results.append(classify(self.tree, X[i]))
            return np.array(results)

    def show(self):
        if self.tree == None:
            raise NotFittedError("Estimator not fitted, call 'fit' first")

        import treePlotter
        treePlotter.createPlot(self.tree)


class NotFittedError(Exception):
    """

    """
    pass
