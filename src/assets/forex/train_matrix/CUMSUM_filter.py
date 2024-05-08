import pandas as pd

def gTEvents(gRaw, h=None):

    h = gRaw['Returns'].std()
    tEvents,sPos,sNeg = [],0,0
    diff = gRaw
    
    for idx, i in diff.iterrows():
        sPos, sNeg = max(0, sPos + i['Returns']), min(0, sNeg + i['Returns'])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(idx)
        elif sPos > h:
            sPos = 0
            tEvents.append(idx)
    return tEvents
    