import pandas as pd
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self):
        self.learning_rate = 0.00001
        self.epoch = 5000

    def predict_(self, x):
        wx = sum([self.weight[j] * x[j] for j in xrange(len(self.weight))])
        return int(wx > 0)

    def train(self, features, labels):
        self.weight = [0.0] * (len(features[0] + 1))
        N = len(features)
        correct_count = 0
        idx = 0

        while idx < self.epoch:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)

            # label 1 -> 1, label 0 -> -1
            y = 2 * labels[index] - 1

            # wx + b
            wx = sum([self.weight[j] * x[j] for j in xrange(len(self.weight))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > N:
                    break
                continue

            # Update weight
            for i in xrange(len(self.weight)-1):
                self.weight[i] += self.learning_rate * (y * x[i])

            # Update bais
                self.weight[len(self.weight)-1] += self.learning_rate * y

    def predict(self, features):
        labels = []

        for feature in features:
            x = list(feature)
            x.append(1.0)

            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':

    print 'Start read data'

    time1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    print raw_data.head()
    print raw_data.info()
    data = raw_data.values

    imgs = data[0::, 1::]

    labels = data[::, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323
    )

    time2 = time.time()

    print 'Read data cost ', time2 - time1, 'second', '\n'

    print 'Start Training'
    p = Perceptron()
    p.train(train_features, train_labels)

    time3 = time.time()
    print 'Training cost ', time3 - time2, 'second', '\n'

    print 'Start Predicting'
    test_predict = p.predict(test_features)
    time4 = time.time()
    print 'Predicting cost ', time4 - time3, 'second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print 'The accruacy score is ', score
