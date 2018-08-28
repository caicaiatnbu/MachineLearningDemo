from decisionTree import decisionTree
X = [[1, 0, 0, 1],
     [1, 0, 0, 2],
     [1, 1, 0, 2],
     [1, 1, 1, 1],
     [1, 0, 0, 1],
     [2, 0, 0, 1],
     [2, 0, 0, 2],
     [2, 1, 1, 2],
     [2, 0, 1, 3],
     [2, 0, 1, 3],
     [3, 0, 1, 3],
     [3, 0, 1, 2],
     [3, 1, 0, 2],
     [3, 1, 0, 3],
     [3, 0, 0, 1]]
y = ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

clf = decisionTree(mode='C4.5')
clf.fit(X, y)
clf.show()
print(clf.predict(X))
