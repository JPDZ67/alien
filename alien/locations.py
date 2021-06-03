import pandas as pd
import numpy as np
import reverse_geocoder as rg


def finds_wierd_values(X):
    """ Finds wierd values in latitude and longitude."""
    X_ = X.copy()
    latitude = X_[[X_['latitude']].str.contains('[A-Za-z]', na=False)]
    longitude = X_[[X_['latitude']].str.contains('[A-Za-z]', na=False)]
    return latitude.index[0], longitude.index[0]


def reverse_geocode(X):
    """Reverse geocodes from latitude and logitude obtains city name.
    Returns a copy of the DataFrame X with 5 columns(latitude,longitude,country, state,city)"""
    X_= X.copy()
    coordinates = list(zip(X_['latitude'], X_['longitude']))
    results = rg.search(coordinates)
    results_df = pd.DataFrame(results)
    X_['country_c'] = results_df['cc']
    X_['city_c'] = results_df['name']
    X_['state_c'] = results_df['admin1']
    X_[['country', 'country_c', 'state', 'state_c', 'city', 'city_c']]
     
    # Checking for nan or '' values in state_c
    X_[(X_['state_c'].notnull()) & (X_['state_c'] == '')].index
        
    # Replace empty values in state_c with the city name
    X_['state_c'] = X_['state_c'].replace({'': np.nan})
    X_['state_c'] = X_['state_c'].fillna(X_['city_c'])

    X_ = X_.drop(columns=['longitude ', 'comments', 'city', 'state', 'country']).rename(columns={"country_c": "country", "city_c": "city", "state_c": "state"})
    return X_[['latitude', 'longitude','country','state','city']]


def us_filter(X):
    """Filters that copy bu US country only"""
    X_ = X.copy()
    return X_[X_['country_c'] == 'US']
