import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        '''
        Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.
        '''

    def fit(self, X, y=None):
        fill = []
        for c in X:
            if X[c].dtype == np.dtype('O'):
                fill.append(X[c].value_counts().index[0])
            else:
                fill.append(X[c].mean())
        self.fill = pd.Series(fill, index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
