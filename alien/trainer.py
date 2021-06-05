import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from termcolor import colored
from alien.utils import compute_rmse
from alien.encoders import TimeFeaturesEncoder
from alien.data import CAT_FEATURES, NUM_FEATURES
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Trainer(object):
    def __init__(self, X, y, **kwargs):
        self.kwargs = kwargs
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get('split', True)
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.15)
        self.pipeline = None

    def get_estimator(self):
        estimator = self.kwargs.get('estimator')
        if estimator == 'Lasso':
            model = Lasso()
        elif estimator == 'Ridge':
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        else:
            model = Lasso()

        return model

    def set_pipeline(self):
        datetime_pipe = make_pipeline(TimeFeaturesEncoder(time_column='year_season'),
                                      OneHotEncoder(handle_unknown='ignore'))

        categorical_pipe = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

        population_pipe = make_pipeline(SimpleImputer())

        numerical_pipe = make_pipeline(StandardScaler())

        feateng_blocks = [
            ('datetime', datetime_pipe, ['year_season']),
            ('categorical', categorical_pipe, CAT_FEATURES),
            ('population', population_pipe, ['population']),
            ('numerical', numerical_pipe, NUM_FEATURES)
        ]

        features_encoder = ColumnTransformer(feateng_blocks, remainder='drop')

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('model', self.get_estimator())
        ])

    def train(self):
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self):
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        if self.split:
            rmse_val = self.compute_rmse(self.X_val, self.y_val, show=True)
            print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_rmse(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)


if __name__ == "__main__":
    params = dict(split=True,
                  estimator='GBM')

    df = pd.read_csv('/Users/juan/code/Polanket/alien/raw_data/merged_df.csv')
    df.drop(columns=['year', 'season'],
            inplace=True,
            errors='ignore')
    y = df['sightings_days']
    X = df.drop('sightings_days', axis=1)
    del df
    t = Trainer(X=X, y=y, **params)
    del X, y

    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
