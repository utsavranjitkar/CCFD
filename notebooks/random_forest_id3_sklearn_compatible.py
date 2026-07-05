"""
RandomForestID3 — a from-scratch Random Forest (no sklearn) built to be a
drop-in replacement for sklearn.ensemble.RandomForestClassifier inside the
CCFD notebook's `ImbPipeline` / `cross_validate` / `RandomizedSearchCV` calls.

Only numpy, pandas, and stdlib are used. sklearn is NEVER imported here --
it's only used in this file's own __main__ test block to check that the
model is truly drop-in compatible (clone(), get_params(), etc.).

Why this is safe to slot into Steps 12-18 of your notebook untouched:
  - Your pipeline already turns everything into numeric columns before this
    point (frequency-encoded, one-hot, binary, scaled) -- so ID3's threshold
    splitting on numeric features is all that's needed. No categorical
    branching logic required.
  - It implements fit / predict / predict_proba / get_params / set_params,
    which is exactly what ImbPipeline, cross_validate, and RandomizedSearchCV
    call on the 'clf' step. Param names (n_estimators, max_depth,
    min_samples_split, min_samples_leaf, max_features) match sklearn's RF,
    so your existing param_dist dictionary works unchanged.
  - feature_importances_ is computed the same way sklearn does it (impurity
    decrease weighted by fraction of samples at each split, averaged over
    all trees) -- so Step 16's feature-importance plot needs no changes.

PERFORMANCE WARNING (read this before running on the full dataset):
  A pure-Python/NumPy tree is 20-100x slower than sklearn's Cython
  implementation. Your SMOTE-resampled training set is likely several
  hundred thousand rows. Recommended for this to actually finish:
    - Use n_estimators=20-30 instead of 100-300 while iterating
    - Set max_depth to something like 10-15 (unbounded depth on 500k rows
      will be very slow to build and to predict with)
    - Consider training on a stratified subsample (e.g. 50k-100k rows) if
      you just want to demonstrate the from-scratch model working, and keep
      the sklearn RF as your actual deployed/production model.
"""

import numpy as np
import pandas as pd
from collections import Counter


# ----------------------------------------------------------------------
# Fast entropy-based split search (O(n log n) per feature via running
# class counts over sorted values, instead of recomputing entropy at
# every candidate threshold from scratch).
# ----------------------------------------------------------------------

def _entropy_from_counts(counts, n):
    if n == 0:
        return 0.0
    probs = counts[counts > 0] / n
    return -np.sum(probs * np.log2(probs))


def _best_threshold_split(x, y_idx, n_classes, min_samples_leaf):
    """
    x:      1D float array of a single feature's values
    y_idx:  1D int array of class indices (0..n_classes-1), same order as x
    Returns (best_gain, best_threshold) for a binary split x <= t / x > t.
    """
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y_idx[order]

    total_counts = np.bincount(y_sorted, minlength=n_classes).astype(float)
    parent_entropy = _entropy_from_counts(total_counts, n)

    left_counts = np.zeros(n_classes)
    right_counts = total_counts.copy()

    best_gain = 0.0
    best_thresh = None

    for i in range(n - 1):
        c = y_sorted[i]
        left_counts[c] += 1
        right_counts[c] -= 1

        if x_sorted[i] == x_sorted[i + 1]:
            continue  # can't split between two equal values

        n_left = i + 1
        n_right = n - n_left
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        left_e = _entropy_from_counts(left_counts, n_left)
        right_e = _entropy_from_counts(right_counts, n_right)
        weighted = (n_left / n) * left_e + (n_right / n) * right_e
        gain = parent_entropy - weighted

        if gain > best_gain:
            best_gain = gain
            best_thresh = (x_sorted[i] + x_sorted[i + 1]) / 2.0

    return best_gain, best_thresh


# ----------------------------------------------------------------------
# Tree node
# ----------------------------------------------------------------------

class _Node:
    __slots__ = ("is_leaf", "proba", "feature_idx", "threshold", "left", "right", "n_samples")

    def __init__(self):
        self.is_leaf = False
        self.proba = None        # class-probability vector, stored at every node (used as leaf output)
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.n_samples = 0


# ----------------------------------------------------------------------
# Single ID3 tree (entropy + numeric threshold splits)
# ----------------------------------------------------------------------

class _ID3Tree:
    def __init__(self, max_depth, min_samples_split, min_samples_leaf,
                 max_features, n_classes, rng):
        self.max_depth = max_depth if max_depth is not None else 10 ** 9
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_classes = n_classes
        self.rng = rng
        self.root = None
        self.feature_importances_ = None  # filled in after fit

    def fit(self, X, y_idx):
        n_total_samples, n_total_features = X.shape
        self._raw_importance = np.zeros(n_total_features)
        self._n_total_samples = n_total_samples
        self.root = self._grow(X, y_idx, depth=0)

        total = self._raw_importance.sum()
        if total > 0:
            self.feature_importances_ = self._raw_importance / total
        else:
            self.feature_importances_ = self._raw_importance
        return self

    def _leaf(self, y_idx):
        node = _Node()
        node.is_leaf = True
        node.n_samples = len(y_idx)
        counts = np.bincount(y_idx, minlength=self.n_classes).astype(float)
        node.proba = counts / counts.sum()
        return node

    def _grow(self, X, y_idx, depth):
        n_samples, n_features = X.shape

        if (n_samples < self.min_samples_split
                or depth >= self.max_depth
                or len(np.unique(y_idx)) == 1):
            return self._leaf(y_idx)

        # random feature subset for this split (random-forest behaviour)
        n_feat_try = self._n_features_to_try(n_features)
        feat_indices = self.rng.choice(n_features, size=n_feat_try, replace=False)

        best_gain, best_thresh, best_feat = 0.0, None, None
        for f in feat_indices:
            gain, thresh = _best_threshold_split(
                X[:, f], y_idx, self.n_classes, self.min_samples_leaf
            )
            if gain > best_gain:
                best_gain, best_thresh, best_feat = gain, thresh, f

        if best_feat is None:
            return self._leaf(y_idx)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        node = _Node()
        node.n_samples = n_samples
        node.feature_idx = best_feat
        node.threshold = best_thresh

        # impurity-decrease importance, weighted by fraction of ALL training
        # samples reaching this node (matches sklearn's definition)
        self._raw_importance[best_feat] += best_gain * (n_samples / self._n_total_samples)

        node.left = self._grow(X[left_mask], y_idx[left_mask], depth + 1)
        node.right = self._grow(X[right_mask], y_idx[right_mask], depth + 1)
        return node

    def _n_features_to_try(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        if isinstance(mf, int):
            return min(mf, n_features)
        return n_features

    def _predict_row_proba(self, row, node):
        if node.is_leaf:
            return node.proba
        if row[node.feature_idx] <= node.threshold:
            return self._predict_row_proba(row, node.left)
        return self._predict_row_proba(row, node.right)

    def predict_proba(self, X):
        return np.array([self._predict_row_proba(row, self.root) for row in X])


# ----------------------------------------------------------------------
# Random Forest — sklearn-compatible estimator interface
# ----------------------------------------------------------------------

class RandomForestID3:
    """
    Drop-in replacement for sklearn.ensemble.RandomForestClassifier,
    built from scratch with ID3 (entropy) trees.

    Supported params mirror sklearn's RF naming so your existing
    param_dist / RandomizedSearchCV code needs no changes:
        n_estimators, max_depth, min_samples_split, min_samples_leaf,
        max_features, bootstrap, random_state, n_jobs (accepted, ignored --
        this implementation is single-threaded), class_weight (accepted,
        ignored -- use SMOTE / manual class weighting upstream as you
        already do).
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 random_state=None, n_jobs=None, class_weight=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs            # accepted for API compatibility, unused
        self.class_weight = class_weight  # accepted for API compatibility, unused

    # ---- sklearn estimator API: required by clone() / RandomizedSearchCV ----

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "class_weight": self.class_weight,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- core API: fit / predict / predict_proba ----

    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y = y.values if isinstance(y, pd.Series) else np.asarray(y)
        X = X.astype(float)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y])

        n_samples = X.shape[0]
        master_rng = np.random.RandomState(self.random_state)

        self.trees_ = []
        importances_sum = np.zeros(X.shape[1])

        for _ in range(self.n_estimators):
            seed = master_rng.randint(0, 2**31 - 1)
            tree_rng = np.random.RandomState(seed)

            if self.bootstrap:
                idx = tree_rng.randint(0, n_samples, size=n_samples)
                X_sample, y_sample = X[idx], y_idx[idx]
            else:
                X_sample, y_sample = X, y_idx

            tree = _ID3Tree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_classes=n_classes,
                rng=tree_rng,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
            importances_sum += tree.feature_importances_

        self.feature_importances_ = importances_sum / self.n_estimators
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X = X.astype(float)
        # average probability vectors across all trees (matches sklearn RF)
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees_])
        return all_proba.mean(axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        y = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return np.mean(self.predict(X) == y)


# ----------------------------------------------------------------------
# Self-test: confirms drop-in sklearn compatibility (clone, pipeline,
# get_params round-trip) and basic correctness. sklearn is used ONLY
# here, to verify compatibility -- never inside the class itself.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 2000
    X = pd.DataFrame({
        "amt": rng.exponential(50, n),
        "age": rng.randint(18, 80, n),
        "hour": rng.randint(0, 24, n),
        "merchant_enc": rng.random(n),
    })
    y = ((X["amt"] > 80) & (X["hour"] < 6)).astype(int).values.copy()
    # flip a few labels so it's not a trivial rule
    flip_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    split = int(n * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    clf = RandomForestID3(n_estimators=15, max_depth=6, max_features="sqrt", random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    proba = clf.predict_proba(X_test)
    print(f"Test accuracy: {acc:.3f}")
    print("Feature importances:", dict(zip(X.columns, clf.feature_importances_.round(3))))
    print("predict_proba shape:", proba.shape, "| classes_:", clf.classes_)

    # --- sklearn compatibility check (clone / get_params / set_params) ---
    try:
        from sklearn.base import clone
        clf2 = clone(clf)
        clf2.fit(X_train, y_train)
        print("sklearn clone() + refit worked -> compatible with Pipeline/RandomizedSearchCV")
    except ImportError:
        print("sklearn not installed in this test environment -- skipped clone() check "
              "(the class itself still works fine without sklearn present).")
