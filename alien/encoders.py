from sklearn.base import BaseEstimator, TransformerMixin
from math import floor


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the season ."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        df = X.copy()

        df[['year', 'season']] = df[self.time_column].str.split('-', expand=True)
        return df[['year', 'season']]


class DurationFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, duration_column="avg_duration(seconds)"):
        self.duration_column = duration_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        df = X.copy()

        df[self.duration_column] = df[self.duration_column].apply(lambda x: floor(x))
        return df[[self.duration_column]]
