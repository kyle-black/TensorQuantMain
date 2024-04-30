import pandas as pd

def gTEvents(gRaw, h=None):

    h = gRaw['Returns'].std()
    tEvents,sPos,sNeg = [],0,0
    diff = gRaw
    
    for i in diff.iterrows():
        sPos,sNeg = max(0,sPos+i[1]['Returns']),min(0,sNeg+i[1]['Returns'])
        if sNeg<-h:
            sNeg=0;tEvents.append(int(i[1]['Date']))
        elif sPos>h:
            sPos=0;tEvents.append(int(i[1]['Date']))
    return tEvents
    