import numpy as np
from sortedcontainers import SortedList
from util import get_data
import matplotlib.pyplot as plt


class KNN(object):

    def __init__(self, k):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        y = np.zeros(len(x))
        for i, test_elem in enumerate(x):
            sl = SortedList()
            for index, elem in enumerate(self.x):
                diff = test_elem - elem
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[index]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[index]))

            votes = {}
            for distance, label in sl:
                votes[label] = votes.get(label, 0) + 1

            max_votes = 0
            max_votes_class = -1
            for label, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = label
        y[i] = max_votes_class
        return y

    def score(self, x, y):
        ypred = self.predict(x)
        return np.mean(y == ypred)


if __name__ == '__main__':
    x, y = get_data(2000)
    limit = 1000
    xtrain, ytrain = x[:limit], y[:limit]
    xtest, ytest = x[limit:], y[limit:]

    train_scores = []
    test_scores = []
    ks = (1, 2, 3, 4, 5)
    for k in ks:
        knn = KNN(k)
        knn.fit(xtrain, ytrain)
        train_scores.append(knn.score(xtrain, ytrain))

        test_scores.append(knn.score(xtest, ytest))

    plt.plot(ks, train_scores, label='train_scores')
    plt.plot(ks, test_scores, label='test_scores')
    plt.legend()
    plt.show()
