import pandas as pd
from datetime import datetime


def datetime_cleanup(df, *main_df):
    """Fixes the datetime column by replacing 24: with 0: and converting to datetime object"""
    df['datetime'] = df['datetime'].apply(lambda x: x.replace('24:', '0:'))
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df.sort_values('datetime', axis=0, ascending=True, inplace=True)
    df = remove_future_dates(df)
    if main_df:
        return get_last_sightings(main_df, df)
    else:
        return df


def remove_future_dates(df):
    today = datetime.today()
    return df[df['datetime'] < today]


def get_last_sightings(main_df, new_df):
    last_date = main_df['datetime'].max()
    return new_df[new_df['datetime'] > last_date]