import numpy as np


class AdaBoostF:
    def __init__(self, base_estimator, n_classes):
        self.estimators_ = [base_estimator]
        self.n_estimators_ = 1
        self.estimator_weights_ = [1]
        self.n_classes = n_classes

    def get_estimators(self):
        return self.estimators_

    def add(self, weak_learner, coeff):
        self.estimators_.append(weak_learner)
        self.estimator_weights_ = np.append(self.estimator_weights_, coeff)

    def get(self, index):
        return self.estimators_[index]

    def replace(self, weak_learner, coeff):
        self.estimators_ = [weak_learner]
        self.estimator_weights_ = np.array([coeff])

        return self

    @staticmethod
    def single_pred(arg):
        i, clf, X = arg
        return i, clf.predict(X)

    def predict(self, X: np.ndarray, pool) -> np.ndarray:
        y_pred = np.zeros((np.shape(X)[0], self.n_classes))
        args = [(i, clf, X) for i, clf in enumerate(self.estimators_)]

        results = pool.map(self.single_pred, args)

        for i, pred in results:
            for j, c in enumerate(pred):
                y_pred[j, int(c)] += self.estimator_weights_[i]
        return np.argmax(y_pred, axis=1)
