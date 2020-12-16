import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score, explained_variance_score
from sklearn.model_selection import ShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import scale
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
import os
import os.path as op
from xgboost import XGBRegressor

from CID import CIDGmm

def mi_importance(perm_imps, covered):
    percentage_uncovered = 1 - covered

    perm_imps_mi = perm_imps / percentage_uncovered

    return perm_imps_mi


cur_path = os.getcwd()

output_path = op.join(cur_path, 'output_data')
input_path = op.join(cur_path, 'input_data')
path_to_data = op.join(input_path, 'intermediate_layer_outputs_cid_test')

# X = np.load(op.join(path_to_data, 'total_inter_outputs_1.npy'))
X = pd.read_csv(op.join(output_path, 'merged_dfs.csv'), index_col=0)
y = np.load(op.join(output_path, 'y_merged_datasets.npy'))
# y = np.load(op.join(path_to_data, 'y_1.npy'))
y = pd.Series(y, index=X.index)

n_bins = 20

q = QuantileTransformer(output_distribution='normal')
# X = q.fit_transform(X)
# X = pd.DataFrame(q.fit_transform(X), columns=X.columns, index=X.index)

cid = CIDGmm(data=X, y=y, n_bins=n_bins, scale_data=True, discretize=True, data_std_threshold=None,
             empirical_mi=False, redund_correction=True, disc_feats=[], cont_feats=np.arange(X.shape[1]),
             kwargs={'max_iter': 500, 'alphas': [0.0001, 0.001, 0.01, 0.1], 'tol': 1e-3})

covered, mi_y = cid.fit(n_samples=1)

np.save('covered_whole_data', covered)
np.save('mi_whole_data', mi_y)

##############################################################################
# Permutation Importances
##############################################################################

original_scores, importances_permutation, importances_clf = \
    permutation_importances(X, y, random_search, splitter, score='roc_auc_score')

np.save(f'perm_imps_{data_str}_n_splits_{n_splits}', importances_permutation)
np.save(f'clf_imps_{data_str}_n_splits_{n_splits}', importances_clf)
np.save(f'original_scores_{data_str}_n_splits_{n_splits}', original_scores)

print(covered)