def duration_cleanup(df):
    data = df.copy()
    data['duration (seconds)'] = data['duration (seconds)'].apply(str_int)
    data['duration (seconds)'] = data['duration (seconds)'].astype('float')
    return data


def str_int(string):
    for s in string.split():
        if 'minutes' or 'minute' in string.split():
            return int(int(s) * 60) if s.isdigit() else None
        elif 'seconds' or 'second' in string.split():
            return int(s) if s.isdigit() else None
        elif 'hours' or 'hour' in string.split():
            return int(int(s) * 3600) if s.isdigit() else None
        else:
            return None
