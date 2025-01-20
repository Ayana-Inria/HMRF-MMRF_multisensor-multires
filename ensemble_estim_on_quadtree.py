import numpy as np
from image import Image
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from rotation_forest import *

def ensemble_estim(imgQuadTree, mapQuadTree, _method):
    randomForestList = []  # list, for each resolution, of list, for each class, of the site statistics
    P = []  # list, for each resolution, of predicted labels

    H = imgQuadTree[0].getHeight()
    W = imgQuadTree[0].getWidth()
    height = H
    width = W
    C = mapQuadTree[0].getNumOfClasses()
    R = len(imgQuadTree)
    B = []
    for r in range(R):
        B.append(imgQuadTree[r].getBands())

    X = [] #for each r: all feature vectors in dataset
    Xtrain = [] #for each r: feature vector that has a label (training)
    Y = [] #for each r: label of training set

    for r in range(R):
        pixelFlat = imgQuadTree[r].pixel.ravel() #put pixel on a single row
        trainPixel = mapQuadTree[r].pixel.reshape(height,width) #reshape to throw away useless band axis

        # -------------------------------------------------
        #|   X is list of matrix [W*H][B]                 |
        #|   where B is the number of bands in layer r    |
        #|   s is the site index                          |
        #|   [s=0     b=0][s=0   b=1]   .   [s=0   b=B-1] | 
        #|   [s=1     b=0]                                |
        #|   .          .                                 | 
        #|   .                                            |
        #|   [s=W*H-1 b=0]   .     .    . [s=W*H-1 b=B-1] |
        # -------------------------------------------------
        X.append(np.empty([width*height, B[r]], dtype = type(pixelFlat[0]))) #matrix that will hold for each row values for a pixel in every band (feature vector)
        
        for i in range (0, width*height): #for every pixel
            X[r][i] = pixelFlat[i::width*height] #store in i row of X values for each band
            # obs: here we are starting from first band and jumping W*H pixel, num of pixel per each band
        
        #find train samples
        m = trainPixel != 0 #get boolean matrix mask
        m = m.reshape(width*height,1) #to column
        numOfTrainSample = np.count_nonzero(m)

        columnM0 = m
        for i in range (B[r]-1): #horizontally add another columnM0 column to match X size
            m = np.hstack((m,columnM0))
        Xtrain.append(X[r][m].reshape(-1, B[r])) #-1 -> infer dimension

        #create empty labels array
        Y.append(np.empty([numOfTrainSample, 1]))
        #fill train labels in Y
        count = 0
        for h in range(height):
            for w in range(width):
                if mapQuadTree[r].pixel[0][h][w] != 0:
                    Y[r][count, 0] = mapQuadTree[r].pixel[0][h][w]
                    count += 1
        #reduce sizes
        width = int(width/2)
        height = int(height/2)

    #---Statistics estimation using Random Forest
    width = W
    height = H
    for r in range(R):
        if _method == "GBRF":
            print('Gradient boosting estimation started')
            clf = GradientBoostingClassifier(max_depth = 10, max_features = "auto", n_estimators=100)
        elif _method == "extraF":
            print('Extra tree estimation started')
            clf = ExtraTreesClassifier(n_jobs=2, random_state=0, n_estimators=100)
        elif _method == "rotF":
            print('Rotation tree estimation started')
            clf = RotationForestClassifier()
        elif _method == "RF":
            print('Random forest estimation started')
            clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=100)
        else:
            print('ERROR: unsupported statistic estimation option: ', _method)
        
        #obs:fit(Xtrain, Y)
        #n_samples : number of feature vectors in training set
        #Xtrain : has dim [n_samples, n_features]
        #Y : has dim [n_samples, n_outputs]
        clf.fit(Xtrain[r], Y[r].ravel())

        #find predicted labels for outputting classification result
        # Apply the classifier we trained to the training set
        P.append(clf.predict(X[r]))

        outSingleRes = [] #list for each class of the statistics for each site

        # predicted probabilities p has dim [n_samples, n_classes]
        predProb = clf.predict_proba(X[r])

        #fill statistics data structure
        for xs in range(C):
            #print('xs ->', xs)
            outSingleRes.append(Image(width, height, 1, pixelType = 'float64')) #create new image for output
            #count = 0
            for h in range(0, height):
                #print('h ->', h)
                for w in range(0, width):
                    #outSingleRes[xs].pixel[0][h][w] = clf.predict_proba(X[r])[w+h*width][xs]
                    outSingleRes[xs].pixel[0][h][w] = predProb[w+h*width][xs]
                    #count += 1
        randomForestList.append(outSingleRes)
        #reduce sizes
        width = int(width/2)
        height = int(height/2)

    return P, randomForestList