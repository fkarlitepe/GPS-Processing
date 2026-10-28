"""Learned ambiguity acceptance test.

A classifier maps the outcome of one integer stage (ratio, bootstrapped success rate and
ADOP-based upper bound, ADOP, dimension, squared norms, WL/N1 stage) to the probability
that the fixed integers are correct. It is trained on GNSS-like Monte Carlo samples with
covariance mis-scaling and unmodelled biases (``ar_simulation``) and its probabilities are
calibrated (isotonic) on an independent sample, so ``ml_threshold`` = 0.99 means "accept
when the estimated probability of a correct fix is at least 99 %".

Candidates are compared by ``ar_benchmark`` at a fixed failure rate; ``DEFAULT_MODEL`` is
the winner of that benchmark (see docs/EKF_ML.md).
"""
import json
from pathlib import Path
import numpy as np
from .ar_simulation import FEATURES, feature_vector, simulate
from .learners import LogisticRegression

DEFAULT_MODEL = 'Logistic regression, quadratic (sklearn)'
_DEFAULT = {}


class NumpyLogistic:
    """The dependency-free v2.1 learner behind the scikit-learn estimator interface."""
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, x, y):
        self.model.fit(x, y)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, x):
        p = self.model.probability(x)
        return np.c_[1-p, p]


def build_model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression as SkLogistic
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVC
    factories = {
        'Logistic regression (numpy, v2.1)': NumpyLogistic,
        'Logistic regression, quadratic (sklearn)': lambda: make_pipeline(
            StandardScaler(), PolynomialFeatures(2), SkLogistic(C=1.0, max_iter=5000)),
        'Random forest': lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0),
        'Gradient boosting (HistGB)': lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
            random_state=0),
        'Neural network (MLP)': lambda: make_pipeline(StandardScaler(), MLPClassifier(
            (32, 16), alpha=1e-3, max_iter=1000, early_stopping=True, random_state=0)),
        'Support vector machine (RBF)': lambda: make_pipeline(StandardScaler(), SVC(C=3.0, probability=True, random_state=0)),
    }
    return factories[name]()


CANDIDATES = ['Logistic regression (numpy, v2.1)', 'Logistic regression, quadratic (sklearn)', 'Random forest',
              'Gradient boosting (HistGB)', 'Neural network (MLP)', 'Support vector machine (RBF)']


class ARValidator:
    def __init__(self, model, info=None):
        self.model, self.info = model, info or {}

    @classmethod
    def train(cls, samples=6000, seed=0, model=DEFAULT_MODEL, calibration=None):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        calibration = calibration or max(samples//3, 500)
        x, y = simulate(samples, seed)
        x_cal, y_cal = simulate(calibration, seed+1000)
        fitted = build_model(model).fit(x, y)
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted), method='isotonic').fit(x_cal, y_cal)
        p = calibrated.predict_proba(x_cal)[:, 1]
        return cls(calibrated, {'model': model, 'samples': len(y), 'calibration_samples': len(y_cal), 'seed': seed,
                                'correct_fraction': float(y.mean()),
                                'training_accuracy': float(np.mean((p >= 0.5) == y_cal)),
                                'features': FEATURES})

    def probability(self, result, stage='n1'):
        return float(self.model.predict_proba(feature_vector(result, stage)[None, :])[0, 1])

    def to_dict(self):
        return {'info': self.info}

    def save(self, path):
        """Pickled with joblib (scikit-learn models); load only files you created yourself."""
        import joblib
        path = Path(path)
        joblib.dump({'model': self.model, 'info': self.info}, path)
        path.with_suffix('.json').write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == '.json':
            raise ValueError(f'{path.name}: v2.1 JSON validators use the old feature set; '
                             'train a new validator (saved as .joblib).')
        import joblib
        data = joblib.load(path)
        if data.get('info', {}).get('features') != FEATURES:
            raise ValueError(f'{path.name} was trained with a different feature set; train a new validator.')
        return cls(data['model'], data['info'])


def default_validator():
    """Deterministic, in-memory cached validator (DEFAULT_MODEL) trained on first use."""
    if 'model' not in _DEFAULT:
        _DEFAULT['model'] = ARValidator.train(samples=4000)
    return _DEFAULT['model']
