

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import causalml
from causalml.metrics import plot_gain, plot_qini, qini_score
from causalml.dataset import synthetic_data
from causalml.inference.tree import plot_dist_tree_leaves_values, get_tree_leaves_mask
from causalml.inference.meta import BaseSRegressor, BaseXRegressor, BaseTRegressor, BaseDRRegressor
from causalml.inference.tree import CausalRandomForestRegressor
from causalml.inference.tree import CausalTreeRegressor


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # Simulate randomized trial: mode=2
    y, X, w, tau, b, e = synthetic_data(mode=2, n=2000, p=20, sigma=5.0)
    df = pd.DataFrame(X)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df.columns = feature_names
    df['outcome'] = y
    df['treatment'] = w
    df['treatment_effect'] = tau

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=111)
    n_test = df_test.shape[0]
    n_train = df_train.shape[0]

    X_train, X_test = df_train[feature_names].values, df_test[feature_names].values
    y_train, y_test = df_train['outcome'].values, df_test['outcome'].values
    treatment_train, treatment_test = df_train['treatment'].values, df_test['treatment'].values

    ctree = CausalTreeRegressor(criterion='causal_mse',
                                control_name=0,
                                min_samples_leaf=200,
                                leaves_groups_cnt=True)
    ctree.fit(X=X_train,
              treatment=treatment_train,
              y=y_train
              )
