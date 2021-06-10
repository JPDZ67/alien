import os
import pandas as pd
from alien.ufo_scrapper import Scrapper
from alien.datetime_cleanup import datetime_cleanup
from alien.duration_cleanup import duration_cleanup

LOCAL_MASTER_PATH = '/Users/juan/code/Polanket/alien/raw_data/sightings.csv'
CAT_FEATURES = ['state', 'shape']
NUM_FEATURES = ['avg_duration(seconds)', 'sightings_cities', 'sightings', 'population']
BUCKET_NAME = 'ufo_sightings'
BUCKET_TRAIN_DATA_PATH = 'data/merged_df.csv'
STORAGE_LOCATION = 'models/alien/model.joblib'


def get_data(ignore_new=True, local=False):
    if local:
        data = pd.read_csv(LOCAL_MASTER_PATH)
    else:
        data = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")

    main_df = clean_data(data)

    if ignore_new:
        return main_df
    else:
        return add_new_data(main_df)


def clean_data(df, *main_df):
    clean = df.copy()
    drop_extra_cols(clean)  # Inplace
    rename_cols(clean)  # Might not be needed with next scrapping
    drop_null_values(clean)  # Inplace
    clean = datetime_cleanup(clean, main_df)
    clean = duration_cleanup(clean)
    return clean


def add_new_data(main_df):
    """Adds new data scrapped from the official website"""
    ufo_scrapper = Scrapper()
    df = ufo_scrapper.run()
    clean_df = clean_data(df, main_df)
    master_df = pd.concat([main_df, clean_df], ignore_index=True)
    return master_df


def drop_extra_cols(df):
    """Remove unused columns in training"""
    df.drop(columns=['latitude', 'longitude ', 'date posted', 'duration (hours/min)', 'comments'],
            inplace=True,
            errors='ignore')


def rename_cols(df):
    """Renames column names to match master df"""
    df.rename(columns={'Datetime': 'datetime',
                       'City': 'city',
                       'State': 'state',
                       'Shape': 'shape',
                       'Duration': 'duration (seconds)',
                       'Summary': 'summary'},
              inplace=True)


def drop_null_values(df):
    """Drops NaN values"""
    df.dropna(inplace=True)


if __name__ == "__main__":
    data = get_data(ignore_new=False)
    print(data.head())
