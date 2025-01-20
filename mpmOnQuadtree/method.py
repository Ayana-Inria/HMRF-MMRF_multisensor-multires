import numpy as np
from math import floor
from hilbertCurve import *


def computeTransProb(C, theta):
    _out = np.zeros((C, C)) + (1 - theta) / (C - 1)
    np.fill_diagonal(_out, theta)
    return _out


def computePrior(mapQuadTree, R, C, resolutionTranProb):
    """ Class prior statistics estimation
        Firstly compute relative frequencies for root layer
        Secondly compute recursively prior for resolutions [0 , ... , R-2]
    """
    prior = []
    for i in range(0, R - 1):  # fill list with empty arrays for resolutions [0 , ... , R-2] to be computed later
        prior.append(np.zeros((C)))
    # count occurrences for each class in mapQuadTree[R-1]
    unique, counts = np.unique(mapQuadTree[R - 1].pixel, return_counts=True)
    # remove first element (0 is for unclassified)
    unique = np.delete(unique, 0)
    counts = np.delete(counts, 0)

    num = counts.sum()  # compute number of samples in the coarsest resolution
    prior.append(counts / num)  # obs: relative freq = num of occurrences / num

    # compute priors for others resolutions
    for i in range(0, R - 1):  # loop on every resolution other than the root
        for xs in range(0, C):  # compute p(xs)
            for xsminus in range(0, C):  # \sum^{xsminus}
                # print('R-2-i ->', R-2-i, 'R-1-i ->', R-1-i,' xs ->', xs, ' xsminus ->', xsminus)
                prior[R - 2 - i][xs] += resolutionTranProb[xs][xsminus] * prior[R - 1 - i][xsminus]
    return prior


def get_trans_contrib(R, C, resolutionTranProb, spatialTranProb, prior, num_neighbours=2):
    """ Compute transition contribution given number of spatial neighbours
        example: num_neighbours = 2: compute delta_{ijkh} that appears in eq.(6) icip. 
        with neighbour set composed by left and upper site
    """
    if num_neighbours == 0:  # only resolution contribution [laferte]
        _contrib = np.zeros((R - 1, C, C), dtype='float32')  # is a tensor of the bottom-up results
        for r in reversed(range(R - 1)):
            for i in range(0, C):  # current site class
                for j in range(0, C):  # father class
                    _contrib[r][i][j] = resolutionTranProb[i][j] * prior[r + 1][j]
                    _contrib[r][i][j] /= prior[r][i]
    elif num_neighbours == 1:
        _contrib = np.zeros((R - 1, C, C, C), dtype='float32')  # is a tensor of the bottom-up results
        for r in reversed(range(R - 1)):
            for i in range(0, C):  # current site class
                for j in range(0, C):  # father class
                    for k in range(0, C):  # left neighbour
                        _contrib[r][i][j][k] = resolutionTranProb[i][j] * spatialTranProb[i][k] * prior[r + 1][j] * \
                                               prior[r][k]
                        _contrib[r][i][j][k] /= (prior[r][i]) ** 2
    elif num_neighbours == 2:
        _contrib = np.zeros((R - 1, C, C, C, C), dtype='float32')  # is a tensor of the bottom-up results
        for r in reversed(range(R - 1)):
            for i in range(0, C):  # current site class
                for j in range(0, C):  # father class
                    for k in range(0, C):  # left neighbour
                        for h in range(0, C):  # up neighbour
                            _contrib[r][i][j][k][h] = resolutionTranProb[i][j] * spatialTranProb[i][k] * \
                                                      spatialTranProb[i][h] * prior[r + 1][j] * prior[r][k] * prior[r][
                                                          h]
                            _contrib[r][i][j][k][h] /= (prior[r][i]) ** 3
    else:
        _contrib = 0
        print('ERROR: get_trand_contrib, unsupported num_neighbours = ', num_neighbours)
    return _contrib


def bottom_up(randomForestList, resolutionTranProb, prior, R, C, H, W, _method):
    """ Execute bottom-up pass for the MPM estimation on the quad-tree
        partialPost -> is p(xs|y_d(s)), where y_d(s) the observations of the descendants
    """
    print('0 layer: compute partialPost')
    partialPost = np.zeros((R, C, H, W), dtype='float32')  # is a tensor of the bottom-up results
    gaussianImageList = randomForestList  # !!!change here when fixing gaussian
    for h in range(0, H):
        for w in range(0, W):
            tot = 0  # normalization factor
            for xs in range(0, C):
                if _method != "gauss":
                    partialPost[0][xs][h][w] = randomForestList[0][xs].pixel[0][h][w]
                else:
                    print('---now unsupported')  # TODO fix gauss: idea: execute the * prior[0][xs] before the bottom-up
                    partialPost[0][xs][h][w] = gaussianImageList[0][xs].pixel[0][h][w] * prior[0][xs]
                if partialPost[0][xs][h][w] == 0:
                    partialPost[0][xs][h][w] = 10 ** -35
                    # print ('partialpost is equal to ->', partialPost[0][xs][h][w])
                tot += partialPost[0][xs][h][w]
            # apply normalization
            for xs in range(0, C):
                partialPost[0][xs][h][w] /= tot

    print('0 layer: compute gamma term')
    # compute gamma(xs) term
    gamma = np.zeros((C, H, W), dtype='float32')  # gamma_s, computed to simplify bottom-up for the upper resolution
    # at each resolution will be recomputed
    for xs in range(0, C):
        for h in range(0, H):
            for w in range(0, W):
                for k in range(0, C):  # sum on classes
                    gamma[xs][h][w] += (partialPost[0][k][h][w] * resolutionTranProb[xs][k]) / prior[0][k]

    width = int(W / 2)
    height = int(H / 2)

    for r in range(1, R):
        print(r, ' layer: compute partialPost')
        for h in range(0, height):
            for w in range(0, width):
                tot = 0  # normalization factor
                for xs in range(0, C):
                    if _method != "gauss":
                        partialPost[r][xs][h][w] = randomForestList[r][xs].pixel[0][h][w]
                    else:
                        partialPost[r][xs][h][w] = gaussianImageList[r][xs].pixel[0][h][w] * prior[r][xs]
                    # product on every child
                    for x in range(0, 2):
                        for y in range(0, 2):
                            partialPost[r][xs][h][w] *= gamma[xs][2 * h + y][2 * w + x]

                    if partialPost[r][xs][h][w] == 0:
                        partialPost[r][xs][h][w] = 10 ** -35

                    tot += partialPost[r][xs][h][w]

                # control if tot is equal to zero
                if tot == 0:
                    print('zero in r ->', r, ' h ->', h, ' w ->', w)

                # apply normalization
                for xs in range(0, C):
                    partialPost[r][xs][h][w] /= tot

        # TODO opt: gamma is usless computed at r=R-1 (check and change)
        print(r, ' layer: compute gamma term')
        gamma = np.zeros((C, height, width), dtype='float32')
        # compute gamma(xs) term at r
        for xs in range(0, C):
            for h in range(0, height):
                for w in range(0, width):
                    for k in range(0, C):  # sum on classes
                        gamma[xs][h][w] += (partialPost[r][k][h][w] * resolutionTranProb[xs][k]) / prior[r][k]
        # reduce sizes
        width = int(width / 2)
        height = int(height / 2)

    return partialPost


def top_down(R, H, W, experiment, partialPost, transitionContrib, transitionContribSingle, partial_folder):
    """ Execute top_down pass for the MPM estimation on the quad-tree
        on the root nothing has to be done
        for every other layer compute the p(xs|y) for every site using eq.(5) icip
        the sites's scanning order vary between top_down versions
    """
    if experiment["method"]["only_portion"]:
        fx = experiment["method"]["coord"]["fx"]
        tx = experiment["method"]["coord"]["tx"]
        fy = experiment["method"]["coord"]["fy"]
        ty = experiment["method"]["coord"]["ty"]
        width = int((tx - fx) / (2**(R-2)))
        height = int((ty - fy) / (2**(R-2)))
        fx = int(fx / (2**(R-2)))
        fy = int(fy / (2**(R-2)))
    else:
        #width = int(W / (R - 1))  # get dimension of layer after the root, equal to W/(2*(R-1))*2
        width = int(W/(2**(R-2)))
        #height = int(H / (R - 1))
        height = int(H/(2**(R-2)))
        # set coord offset to 0
        fx = 0
        fy = 0

    alreadyChanged = np.zeros((R, H, W), dtype='bool')

    _version = experiment["method"]["top_down"]  # top-down version chosen

    for r in reversed(range(R - 1)):  # complete top down
        # for r in reversed(range(1, R-1)): #only r = 1
        # for r in reversed(range(R-2)): #only r = 0 !!! if you do change dims
        # increase sizes
        # width = int(width*2)
        # height = int(height*2)
        # fx = int(fx*2)
        # fy = int(fy*2)

        # load previous partialPost file
        # print('loading partialPost at GBRF_top-down_r_0_pntTop_500.npy')
        # partialPost = np.load("datasets/alessandria_power_two/partial_post/GBRF_top-down_r_0_pntTop_500.npy")

        print('--R-- -> --', r, '--')

        if _version == "SMMRF":
            partialPost = spiral(height, width, r, fx, fy, partialPost, alreadyChanged, transitionContrib, experiment,
                                 partial_folder)
        elif _version == "hilZZ":
            partialPost = hilZZ(height, width, r, fx, fy, partialPost, alreadyChanged, transitionContribSingle)
        else:
            print('ERROR: unsupported top-down version ', _version)

        # increase sizes
        width = int(width * 2)
        height = int(height * 2)
        fx = int(fx * 2)
        fy = int(fy * 2)

        # save partialPost after iteration r
        print('saving partialPost after top-down, iteration r = ', r)
        _tmp_file = partial_folder + '/' + str(experiment["statistic_estimation"]["method"]) + '_top-down_r_' + str(r)
        np.save(_tmp_file, partialPost)

    return partialPost


def spiral(height, width, r, fx, fy, partialPost, alreadyChanged, transitionContrib, experiment, partial_folder):
    """ Spiral scanning
    # Scanning is done accordingly to Jurse (Ihsen, Moser, Zerubia)
    # 
    # spiralState: is one of the four way of scanning depending on directions
    # pntTop and pntBottom: hold memory of the y indexes while descending and increasing
    #
    # Boundaries: not computing post MPM for pixel in the first or last row and first or last column
    #   [ X ][ X ][...][ X ]
    #   [ X ] ...  ... [ X ]
    #   [ : ] ...  ... [ : ]
    #   [ : ] ...  ... [ : ]
    #   [ X ][ X ][...][ X ]
    """
    spiralState = 1
    pntTop = 1 + fy  # fy offset is for skipping part of images
    pntBottom = height - 2 + fy
    h = -1  # invalid value to be printed at the beginning

    # Example of initialization if you want to start from another point in the image (ignore)
    # spiralState = 4
    # pntTop = 500
    # pntBottom = 522
    # h = 499

    # top-down in a res
    # TODO fix for first and last row in the case of odd num of lines..
    # while(pntTop!=height):
    while (pntTop != height - 2 + fy):

        # saving state every 200 lines
        if pntTop % 200 == 0:
            print('saving partialPost top-down, iteration r = ', r, ', pntTop = ', pntTop)
            _tmp_file = partial_folder + '/' + str(experiment["statistic_estimation"]["method"]) + '_top-down_r_' + str(
                r) + '_pntTop_' + str(pntTop)
            np.save(_tmp_file, partialPost)
            # this it to print scanning variables
            # if pntTop == 500:
            #    print('---spiralState -> ', spiralState)
            #    print('---pntTop -> ', pntTop)
            #    print('---pntBottom -> ', pntBottom)
            #    print('---h -> ', h)

        if h != -1:
            print('h -> ', h)

        if spiralState == 1:
            for w in reversed(range(1 + fx, width - 1 + fx)):
                h = pntBottom
                spiralScanSite(h, w, r, spiralState, partialPost, alreadyChanged, transitionContrib)
            pntBottom -= 1
            # change state to b
            spiralState = 2

        elif spiralState == 2:
            for w in range(1 + fx, width - 1 + fx):
                h = pntBottom
                spiralScanSite(h, w, r, spiralState, partialPost, alreadyChanged, transitionContrib)
            pntBottom -= 1
            # change state to c
            spiralState = 3

        elif spiralState == 3:
            for w in range(1 + fx, width - 1 + fx):
                h = pntTop
                spiralScanSite(h, w, r, spiralState, partialPost, alreadyChanged, transitionContrib)
            pntTop += 1
            # change state to d
            spiralState = 4

        elif spiralState == 4:
            for w in reversed(range(1 + fx, width - 1 + fx)):
                h = pntTop
                spiralScanSite(h, w, r, spiralState, partialPost, alreadyChanged, transitionContrib)
            pntTop += 1
            # change state to a
            spiralState = 1

    return partialPost


def hilZZ(height, width, r, fx, fy, partialPost, alreadyChanged, transitionContribSingle):
    """ Hilbert curve and zig zag scanning
    # p -> construction number
    # D -> 2^p, size of each dimensions (in 2D: DxD square)
    # L -> D^2, number of pixels
    # N -> number of dimension, 2 in this case
    """
    # TODO insert a check that we use a power two sized dataset
    # otherwise: print('Hilbert curve scanning must run on a power two dataset')
    if height != width:
        raise Exception("Only square power two images are currently supported for hilbert scanning")

    print("----> ", height)
    print("----> ", width)

    # find p: construction number of curve computation
    p = 0
    _width = width
    while _width != 1:
        _width = int(_width / 2)
        # print(_width)
        p += 1
    # debug
    # print('p -> ', p)
    N = 2
    # create hilbert curve
    hilb = HilbertCurve(p, N)

    # ---first hilbert
    axe_hil = 0  # orientation of the hilber curve

    # starting values
    # w_oldold = hilb.coordinates_from_distance(0, orientation= axe_hil)[0]
    # h_oldold = hilb.coordinates_from_distance(0, orientation= axe_hil)[1]
    w_old = hilb.coordinates_from_distance(1, orientation=axe_hil)[0]
    h_old = hilb.coordinates_from_distance(1, orientation=axe_hil)[1]

    # scan every pixel in the curve
    # obs: first pixel is discarted for having at least one past in every pixel
    for ii in range(1, hilb.L):  # ii is the index in the hilbert curve
        # progress number
        # prog = int((ii/L)*100)
        # prog_old = 100

        # if prog != prog_old:
        #	print(prog, '%')
        #	prog_old = int(prog)

        if ii % 600 == 0:
            prog = int((ii / hilb.L) * 100)
            print(prog, '%')

        # 2D current coordinates
        w = hilb.coordinates_from_distance(ii, orientation=axe_hil)[0]
        h = hilb.coordinates_from_distance(ii, orientation=axe_hil)[1]

        # print('h ->', h, end=' ')
        # print('w ->', w)

        scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                            h_old=h_old,
                            w_old=w_old)

        '''
        #understand if there are two or only one nighbour
        #compute distance between current point and oldold point
        dist = sqrt((w-w_oldold)**2+(h-h_oldold)**2)

        if dist > 1.9: #one neighbour
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                h_old = h_old,
                w_old = w_old)
        else: #two neighbour
            scanSite(h, w, r, partialPost, alreadyChanged, transitionContrib, 
                    hilbertScan = 1,
                    h_old = h_old,
                    w_old = w_old,
                    h_oldold=h_oldold,
                    w_oldold=w_oldold)
        '''
        # update old values
        # w_oldold = w_old
        # h_oldold = h_old
        w_old = w
        h_old = h

    # ---second hilbert
    axe_hil = 1  # orientation of the hilber curve

    # starting values
    # w_oldold = hilb.coordinates_from_distance(0, orientation= axe_hil)[0]
    # h_oldold = hilb.coordinates_from_distance(0, orientation= axe_hil)[1]
    w_old = hilb.coordinates_from_distance(1, orientation=axe_hil)[0]
    h_old = hilb.coordinates_from_distance(1, orientation=axe_hil)[1]

    # scan every pixel in the curve
    # obs: first two pixel are discarted for having past (2 oldest) in every pixel
    for ii in range(1, hilb.L):  # ii is the index in the hilbert curve

        if ii % 600 == 0:
            prog = int((ii / hilb.L) * 100)
            print(prog, '%')

        # 2D current coordinates
        w = hilb.coordinates_from_distance(ii, orientation=axe_hil)[0]
        h = hilb.coordinates_from_distance(ii, orientation=axe_hil)[1]

        # print('h ->', h, end=' ')
        # print('w ->', w)

        scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                            h_old=h_old,
                            w_old=w_old)

        '''
        #understand if there are two or only one nighbour
        #compute distance between current point and oldold point
        dist = sqrt((w-w_oldold)**2+(h-h_oldold)**2)

        if dist > 1.9: #one neighbour
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                h_old = h_old,
                w_old = w_old)
        else: #two neighbour
            scanSite(h, w, r, partialPost, alreadyChanged, transitionContrib, 
                    hilbertScan = 1,
                    h_old = h_old,
                    w_old = w_old,
                    h_oldold=h_oldold,
                    w_oldold=w_oldold)
        '''
        # update old values
        # w_oldold = w_old
        # h_oldold = h_old
        w_old = w
        h_old = h

    # ---first zigzag
    # from top left
    print('first zigzag')
    h_old = 0
    w_old = 0
    h = 0
    w = 1

    countProg = 0
    while (h != height - 1 or w != width - 1):
        countProg += 1
        if countProg % 700 == 0:
            print(int((countProg / (width * height)) * 100), '%')

        axe = (h + w) % 2  # direction
        if axe == 0:
            # save old values
            h_old = h
            w_old = w
            # going up
            w += 1
            h -= 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)
        else:
            # save old values
            h_old = h
            w_old = w
            # going down
            w -= 1
            h += 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)

        # check if we are on horizontal border
        if h == 0 or h == height - 1:
            # save old values
            h_old = h
            w_old = w
            w += 1
            # dont scan on border
        # check if we are on vertical border
        elif w == 0 or w == width - 1:
            # save old values
            h_old = h
            w_old = w
            h += 1
            # dont scan on border

    # ---second zigzag
    # from bottom right
    print('second zigzag')
    h_old = height - 1
    w_old = width - 1
    h = height - 1
    w = width - 2

    countProg = 0
    while (h != 0 or w != 0):
        countProg += 1
        if countProg % 700 == 0:
            print(int((countProg / (width * height)) * 100), '%')

        axe = (h + w) % 2  # direction
        if axe == 0:
            # save old values
            h_old = h
            w_old = w
            # going down
            w -= 1
            h += 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)
        else:
            # save old values
            h_old = h
            w_old = w
            # going up
            w += 1
            h -= 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)

        # check if we are on horizontal border
        if h == 0 or h == height - 1:
            # save old values
            h_old = h
            w_old = w
            w -= 1
            # dont do scan on border
        # check if we are on vertical border
        elif w == 0 or w == width - 1:
            # save old values
            h_old = h
            w_old = w
            h -= 1
            # dont do scan on border

    # ---third zigzag
    # from top right
    print('third zigzag')
    h_old = 0
    w_old = width - 1
    h = 1
    w = width - 1

    countProg = 0
    while (h != height - 1 or w != 0):
        countProg += 1
        if countProg % 700 == 0:
            print(int((countProg / (width * height)) * 100), '%')

        axe = (h + w) % 2  # direction
        if axe == 0:
            # save old values
            h_old = h
            w_old = w
            # going up
            w -= 1
            h -= 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)
        else:
            # save old values
            h_old = h
            w_old = w
            # going down
            w += 1
            h += 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)

        # check if we are on vertical border
        if w == 0 or w == width - 1:
            # save old values
            h_old = h
            w_old = w
            h += 1
            # dont do scan on border
        # check if we are on horizontal border
        elif h == 0 or h == height - 1:
            # save old values
            h_old = h
            w_old = w
            w -= 1
            # dont do scan on border

    # ---fourth zigzag
    # from bottom left
    print('fourth zigzag')
    h_old = height - 1
    w_old = 0
    h = height - 2
    w = 0

    countProg = 0
    while (h != 0 or w != width - 1):
        countProg += 1
        if countProg % 700 == 0:
            print(int((countProg / (width * height)) * 100), '%')

        axe = (h + w) % 2  # direction
        if axe == 0:
            # save old values
            h_old = h
            w_old = w
            # going down
            w += 1
            h += 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)
        else:
            # save old values
            h_old = h
            w_old = w
            # going up
            w -= 1
            h -= 1
            # do scan
            scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                                h_old=h_old,
                                w_old=w_old)

        # check if we are on vertical border
        if w == 0 or w == width - 1:
            # save old values
            h_old = h
            w_old = w
            h -= 1
            # dont do scan on border
        # check if we are on horizontal border
        elif h == 0 or h == height - 1:
            # save old values
            h_old = h
            w_old = w
            w += 1
            # dont do scan on border

    return partialPost


def spiralScanSite(h, w, r, spiralState, partialPost, alreadyChanged, transitionContrib):
    """ Scan site for implementing the SMMRF with 2 neighbours on the quadtree structure
        spiral traversing states: spiralState
            1: bottom to top, right to left
            2: bottom to top, left to right
            3: top to bottom, left to right
            4: top to bottom, right to left
    """
    C = partialPost.shape[1]
    _sum = []

    tot = np.zeros((C, C, C))
    for xsminus in range(0, C):
        for xsLeft in range(0, C):
            tot[xsminus, xsLeft, :] = np.sum(np.multiply(partialPost[r, :, h, w], transitionContrib[r, :, xsminus, xsLeft, :].T), axis=1)  # broadcasting is supposed to happen here
    tot[tot == 0] = 10 ** -35


    for i in range(0, C):
        _sum.append(0)
        for xsminus in range(0, C):
            for xsLeft in range(0, C):
                for xsTop in range(0, C):
                    # apply normalization and compute a term of the sum on j,k,h (xsminus,xsLeft,xsTop)
                    # obs: start with assignment for addendo
                    addendo = partialPost[r][i][h][w] * transitionContrib[r][i][xsminus][xsLeft][xsTop] / tot[xsminus, xsLeft, xsTop]
                    addendo *= partialPost[r + 1][xsminus][floor(h / 2)][floor(w / 2)]
                    if spiralState == 1:
                        addendo *= partialPost[r][xsLeft][h][w + 1]
                        addendo *= partialPost[r][xsTop][h + 1][w]
                    elif spiralState == 2:
                        addendo *= partialPost[r][xsLeft][h][w - 1]
                        addendo *= partialPost[r][xsTop][h + 1][w]
                    elif spiralState == 3:
                        addendo *= partialPost[r][xsLeft][h][w - 1]
                        addendo *= partialPost[r][xsTop][h - 1][w]
                    elif spiralState == 4:
                        addendo *= partialPost[r][xsLeft][h][w + 1]
                        addendo *= partialPost[r][xsTop][h - 1][w]
                    _sum[i] += addendo

    # set to valid probabilities _sum (sum to 1)
    # then compute the sum on classes and divide
    _normSum = 0
    for i in range(0, C):
        _normSum += _sum[i]
    for i in range(0, C):
        _sum[i] /= _normSum

    if not alreadyChanged[r][h][w]:  # first time this site is scanned
        for i in range(0, C):
            partialPost[r][i][h][w] = _sum[i]
        alreadyChanged[r][h][w] = True
    else:  # already passed here
        for i in range(0, C):
            partialPost[r][i][h][w] += _sum[i]
        # set again to valid probabilities (sum to 1)
        # then compute the sum on classes and divide
        _normSum = 0
        for i in range(0, C):
            _normSum += partialPost[r][i][h][w]
        for i in range(0, C):
            partialPost[r][i][h][w] /= _normSum


def scanSiteSingleNeigh(h, w, r, partialPost, alreadyChanged, transitionContribSingle,
                        h_old=None,
                        w_old=None):
    C = partialPost.shape[1]

    _sum = []

    tot = np.zeros((C, C))
    for xsminus in range(0, C):
        tot[xsminus, :] = np.sum(np.multiply(partialPost[r, :, h, w], transitionContribSingle[r, :, xsminus, :].T), axis=1)  # broadcasting is supposed to happen here
    tot[tot == 0] = 10 ** -35


    for i in range(0, C):
        _sum.append(0)
        for xsminus in range(0, C):
            for xsLeft in range(0, C):

                #tot = 0  # normalization factor
                #for I in range(0, C):
                #    tot += partialPost[r][I][h][w] * transitionContribSingle[r][I][xsminus][xsLeft]
                #if tot == 0:
                #    tot = 10 ** -35

                # apply normalization and compute a term of the sum on j,k,h (xsminus,xsLeft,xsTop)
                # obs: start with assignment for addendo
                addendo = partialPost[r][i][h][w] * transitionContribSingle[r][i][xsminus][xsLeft] / tot[xsminus, xsLeft]
                addendo *= partialPost[r + 1][xsminus][floor(h / 2)][floor(w / 2)]
                # single neighbour
                addendo *= partialPost[r][xsLeft][h_old][w_old]

                _sum[i] += addendo

    # set to valid probabilities _sum (sum to 1)
    # then compute the sum on classes and divide
    _normSum = 0
    for i in range(0, C):
        _normSum += _sum[i]
    for i in range(0, C):
        _sum[i] /= _normSum

    if not alreadyChanged[r][h][w]:  # first time this site is scanned
        for i in range(0, C):
            partialPost[r][i][h][w] = _sum[i]
        alreadyChanged[r][h][w] = True
    else:  # already passed here
        for i in range(0, C):
            partialPost[r][i][h][w] += _sum[i]
        # set again to valid probabilities (sum to 1)
        # then compute the sum on classes and divide
        _normSum = 0
        for i in range(0, C):
            _normSum += partialPost[r][i][h][w]
        for i in range(0, C):
            partialPost[r][i][h][w] /= _normSum
