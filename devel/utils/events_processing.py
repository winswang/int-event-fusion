import numpy as np
# function for loading events
def loadEventsWithinTimeWindow(eventFileString, timeStart, timeEnd):
    eventsTimestamp = []
    eventsXYP = []
    for line in open(eventFileString, 'r'):
        txyp = line[:-1].split(' ')
        timestamp = float(txyp[0])
        if timestamp >= timeStart and timestamp <= timeEnd:
            eventsTimestamp.append(timestamp)
            eventsXYP.append([int(txyp[1]), int(txyp[2]), int(txyp[3])*2-1])
    print("# of events:", np.shape(eventsTimestamp))
    return np.array(eventsTimestamp), np.array(eventsXYP)  

def findTimeWindowFromId(imagesFileString, imgFileList, imgPairId):
    for line in open(imagesFileString, 'r'):
        lineList = line[:-1].split(' ')
        if lineList[1] == 'images/' + imgFileList[imgPairId[0]]:
            timeStart = float(lineList[0])
        if lineList[1] == 'images/' + imgFileList[imgPairId[1]]:
            timeEnd = float(lineList[0])
    return timeStart, timeEnd

def binEventFrames(eventsTime, eventsXYP, method = 0, frameNum = 1):
    
    if method == 0: # bin events evenly
        eventFrames = np.zeros((frameNum, 180, 240))
        tMin = np.min(eventsTime)
        tMax = np.max(eventsTime)
        if frameNum == 1:
            for i, _ in enumerate(eventsTime):
                eventFrames[0, eventsXYP[i][1], eventsXYP[i][0]] += eventsXYP[i][2]
        else:
            tUnit = (tMax - tMin) / frameNum
            for i, time in enumerate(eventsTime):
                timeToFrameNum = int((time - tMin) // (tUnit+1e-10))
                eventFrames[timeToFrameNum, eventsXYP[i][1], eventsXYP[i][0]] += eventsXYP[i][2]
    elif method == 1: # dynamic binning, binned frames have only 3 values (-1, 0, 1)
        for i in range(len(eventsTime)):
            if i == 0:
                eventFrames = np.zeros((1, 180, 240))
                eventFrames[0, eventsXYP[i][1], eventsXYP[i][0]] = eventsXYP[i][2]
            elif eventFrames[-1, eventsXYP[i][1], eventsXYP[i][0]] != 0:
                eventFrames = np.concatenate((eventFrames, np.zeros((1, 180, 240))), axis = 0)
                eventFrames[-1, eventsXYP[i][1], eventsXYP[i][0]] = eventsXYP[i][2]
            else:
                eventFrames[-1, eventsXYP[i][1], eventsXYP[i][0]] = eventsXYP[i][2]
        print("Binned evf shape:", np.shape(eventFrames))
        
    return eventFrames

def plot_color_evf(evf):
    dim1 = np.size(evf, 0)
    dim2 = np.size(evf, 1)
    neg_c = np.array([212, 20, 90])/255.
    pos_c = np.array([63, 169, 245])/255.
    canvas = np.ones((dim1, dim2, 3))*0.5
    for i in range(dim1):
        for j in range(dim2):
            if evf[i,j] == 1:
                canvas[i,j,:] = pos_c
            elif evf[i,j] == -1:
                canvas[i,j,:] = neg_c
    return canvas

def overlayEventsOnImg(gray_img, eventsXYP):
    # gray_img should be in shape (y, x)
    neg_c = np.array([212, 20, 90])/255.
    pos_c = np.array([63, 169, 245])/255.
    img_ = np.expand_dims(gray_img, 2)
    canvas = np.concatenate((img_, img_, img_), axis = 2)
    for i, xyp in enumerate(eventsXYP):
        if xyp[2] < 0:
            canvas[xyp[1], xyp[0], :] = neg_c
        else:
            canvas[xyp[1], xyp[0], :] = pos_c
    return canvas

def binningStats(eventTxtFile):
    currentFrame = np.zeros((180, 240))
    startTime = 0.0
    durationArray = []
    eventsNumArray = []
    for line in open(eventTxtFile, 'r'):
        txyp = line[:-1].split(' ')
        timestamp = float(txyp[0])
        x = int(txyp[1])
        y = int(txyp[2])
        
        if currentFrame[y, x] == 1:
            durationArray.append(timestamp - startTime)
            startTime = timestamp
            eventsNumArray.append(np.sum(currentFrame))
        else:
            currentFrame[y, x] += 1
    return durationArray, eventsNumArray
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    