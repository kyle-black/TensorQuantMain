import pandas as pd

def gTEvents(gRaw, h):
    """
    This function finds events in a time series data where the cumulative percentage change
    exceeds 2 times the threshold h (both positive and negative).

    :param gRaw: DataFrame containing time series data with 'pct_change' column
    :param h: Threshold value for detecting significant events
    :return: List of timestamps where significant events are detected
    """
    tEvents = []  # List to store timestamps of detected events
    sPos, sNeg = 0, 0  # Initialize cumulative positive and negative changes

    # Iterate through each row in the DataFrame
    for idx, i in gRaw.iterrows():
        sPos, sNeg = max(0, sPos + i['pct_change']), min(0, sNeg + i['pct_change'])
        if sNeg < -2 * h:
            sNeg = 0
            tEvents.append(idx)
        elif sPos > 2 * h:
            sPos = 0
            tEvents.append(idx)

    return tEvents


# Sample data
