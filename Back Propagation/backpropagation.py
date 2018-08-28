import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class neuralNetwork:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        self.ai = [1.] * self.ni
        self.ah = [1.] * self.nh
        self.ao = [1.] * self.no

        self.wi = np.random.random((self.ni, self.nh)) ** 2 - 1
        self.wo = np.random.random((self.nh, self.no)) ** 2 - 1


    def feedForwad(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('error!!!')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    def backPropagation(self, y, learningRate):
        if len(y) != self.no:
            raise ValueError('error!!!')

        delta_k = [0.] * self.no
        for k in range(self.no):
            ek = y[k] - self.ao[k]
            delta_k[k] = ek * dsigmoid(self.ao[k])

        delta_j = [0.] * self.nh
        for j in range(self.nh):
            ej = 0.
            for k in range(self.no):
                ej = ej + delta_k[k] * self.wo[j][k]
            delta_j[j] = dsigmoid(self.ah[j]) * ej

        # Update weight of input layer
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = self.wo[j][k] + learningRate * delta_k[k] * self.ah[j]

        # update weight of output layer
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = self.wi[i][j] + learningRate * delta_j[j] * self.ai[i]

        # 计算误差
        error = 0.0
        for k in range(len(y)):
            error = error + 0.5 * (y[k] - self.ao[k]) ** 2
        return error

    def test(self, pattern):
        print('-----test-----')
        for p in pattern:
            print(p[0], '->', self.feedForwad(p[0]))

    def getWeight(self):
        print('-----weight of input layer-----')
        for i in range(self.ni):
            print(self.wi[i])
        print('-----weight of output layer-----')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, pattern, epoch=10000, alpha=0.5):
        for i in range(epoch):
            error = 0.
            for p in pattern:
                inputs = p[0]
                y = p[1]
                self.feedForwad(inputs)
                error = error + self.backPropagation(y, alpha)

            if i % 100 == 0:
                print('[epoch:%d] error %-.5f' % (i,error))

def demo():
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    n = neuralNetwork(2, 4, 1)
    n.train(pat)
    n.test(pat)
    n.getWeight()

if __name__ == '__main__':
    demo()

