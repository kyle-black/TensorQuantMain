import pandas as pd

def gTEvents(gRaw, h=None):

    h = gRaw['Close'].pct_change().std()
    tEvents,sPos,sNeg = [],0,0
    diff = gRaw
    
    for idx, i in diff.iterrows():
        sPos, sNeg = max(0, sPos + i['pct_change']), min(0, sNeg + i['pct_change'])
        if sNeg < -(2*h):
            sNeg = 0
            tEvents.append(idx)
        elif sPos > (2*h):
            sPos = 0
            tEvents.append(idx)
    return tEvents



