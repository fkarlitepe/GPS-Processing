"""Dependency-free learners: gradient-boosted regression trees and L2 logistic regression.

Both serialize to plain dictionaries so trained models can be stored as JSON next to
results and reloaded without pickle.
"""
import numpy as np


def _fit_tree(x, y, depth, min_leaf):
    if depth == 0 or len(y) < 2*min_leaf:
        return {'value': float(y.mean())}
    best = None
    total, count = y.sum(), len(y)
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind='stable')
        xs, ys = x[order, feature], y[order]
        left = np.cumsum(ys)[min_leaf-1:count-min_leaf]
        n_left = np.arange(min_leaf, count-min_leaf+1)
        valid = xs[min_leaf-1:count-min_leaf] < xs[min_leaf:count-min_leaf+1]
        if not valid.any():
            continue
        gain = left**2/n_left+(total-left)**2/(count-n_left)
        gain[~valid] = -np.inf
        i = int(np.argmax(gain))
        if best is None or gain[i] > best[0]:
            threshold = (xs[min_leaf-1+i]+xs[min_leaf+i])/2
            best = (gain[i], feature, threshold)
    if best is None or best[0] <= total**2/count+1e-12:
        return {'value': float(y.mean())}
    _, feature, threshold = best
    mask = x[:, feature] <= threshold
    return {'feature': int(feature), 'threshold': float(threshold),
            'left': _fit_tree(x[mask], y[mask], depth-1, min_leaf),
            'right': _fit_tree(x[~mask], y[~mask], depth-1, min_leaf)}


def _predict_tree(node, x):
    out = np.empty(len(x))
    stack = [(node, np.arange(len(x)))]
    while stack:
        node, rows = stack.pop()
        if 'value' in node:
            out[rows] = node['value']
            continue
        mask = x[rows, node['feature']] <= node['threshold']
        stack += [(node['left'], rows[mask]), (node['right'], rows[~mask])]
    return out


class GradientBoostingRegressor:
    """Least-squares gradient boosting (Friedman 2001) with depth-limited trees."""

    def __init__(self, n_estimators=60, learning_rate=0.1, max_depth=3, min_samples_leaf=20,
                 max_samples=20000, seed=0):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf
        self.max_samples, self.seed = max_samples, seed
        self.base, self.trees = 0.0, []

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(y) > self.max_samples:
            keep = np.random.default_rng(self.seed).choice(len(y), self.max_samples, replace=False)
            x, y = x[keep], y[keep]
        leaf = max(1, min(self.min_samples_leaf, len(y)//4))
        self.base = float(y.mean())
        prediction = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = _fit_tree(x, y-prediction, self.max_depth, leaf)
            self.trees.append(tree)
            prediction += self.learning_rate*_predict_tree(tree, x)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        prediction = np.full(len(x), self.base)
        for tree in self.trees:
            prediction += self.learning_rate*_predict_tree(tree, x)
        return prediction

    def to_dict(self):
        return {'type': 'gbrt', 'base': self.base, 'learning_rate': self.learning_rate, 'trees': self.trees}

    @classmethod
    def from_dict(cls, data):
        model = cls(learning_rate=data['learning_rate'])
        model.base, model.trees = data['base'], data['trees']
        return model


class LogisticRegression:
    """L2-regularized logistic regression fitted by Newton/IRLS on standardized features."""

    def __init__(self, l2=1e-2):
        self.l2 = l2
        self.mean = self.scale = self.weights = None

    def _design(self, x):
        return np.c_[np.ones(len(x)), (np.asarray(x, dtype=float)-self.mean)/self.scale]

    def fit(self, x, y, iterations=50):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.mean, self.scale = x.mean(0), np.where(x.std(0) > 0, x.std(0), 1.0)
        a = self._design(x)
        w = np.zeros(a.shape[1])
        penalty = np.eye(len(w))*self.l2*len(y); penalty[0, 0] = 0
        for _ in range(iterations):
            p = 1/(1+np.exp(-np.clip(a@w, -30, 30)))
            hessian = a.T@(a*(p*(1-p))[:, None])+penalty
            step = np.linalg.solve(hessian, a.T@(p-y)+penalty@w)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self

    def probability(self, x):
        return 1/(1+np.exp(-np.clip(self._design(np.atleast_2d(x))@self.weights, -30, 30)))

    def to_dict(self):
        return {'type': 'logistic', 'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'weights': self.weights.tolist(), 'l2': self.l2}

    @classmethod
    def from_dict(cls, data):
        model = cls(data.get('l2', 1e-2))
        model.mean, model.scale, model.weights = (np.array(data[k]) for k in ('mean', 'scale', 'weights'))
        return model
