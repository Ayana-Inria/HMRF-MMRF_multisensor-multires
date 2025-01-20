import numpy as np
import pathlib
import os
import datetime
import time

from math import isnan
from image import Image
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def set_output_location(experiment_name):
    output_path = 'output/' + str(experiment_name)
    # check if directory for output exist, if not creates it
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # write in file the starting time
    print('>>>Starting the experiment:', experiment_name)
    f = open(output_path + '/result.txt', "a+")
    f.write('>>>Starting the experiment: ' + experiment_name + '\n')
    f.write('Results at time: ' + str(datetime.datetime.now()) + '\n\r')
    f.close()  # close file
    return output_path


def img_from_data(prob, H, W, R, output_path, img_name, _from='prob'):
    """ Create images of predicted labels on every resolution
        prob -> probability of each class, format expected [R][C][H][W]
        label -> label already given
    """
    _result_img = []  # list of result images
    height = H
    width = W
    if _from == 'prob':
        C = prob.shape[1]

    for r in range(R):
        imageResult = Image(width, height, pixelType='uint8')  # result of classification image
        if _from == 'prob':
            # label choosed is argmax on xs of partialPost
            # obs:
            #	done in a numpy way would be
            #	imageResult = np.argmax(partialPost, axis=1)
            for h in range(height):
                for w in range(width):
                    label = 0
                    maxValue = 0
                    for xs in range(C):
                        if isnan(prob[r][xs][h][w]):
                            print('WARNING: nan found')
                        else:
                            if prob[r][xs][h][w] > maxValue:
                                label = xs + 1
                                maxValue = prob[r][xs][h][w]
                    # store label
                    imageResult.pixel[0][h][w] = label

                    # set to background the first row and first column
                    if h == 0 or w == 0 or h == height - 1 or w == width - 1:
                        imageResult.pixel[0][h][w] = 0
        elif _from == 'label':
            PSingleRes = prob[r].reshape(height, width)  # reshape the array to have the right dimension
            # fill result image and compute accuracy
            for h in range(0, height):
                for w in range(0, width):
                    imageResult.pixel[0][h][w] = PSingleRes[h][w]
        else:
            print('ERROR: unsupported option _from: ', _from)

        imageResult.store(output_path + '/' + str(img_name) + '_r_' + str(r) + '.raw', 1)

        # --- taking care of the .hdr file
        # if .hdr file not exist create it
        if not os.path.isfile(output_path + '/' + str(img_name) + '_r_' + str(r) + '.hdr'):
            f = open(output_path + '/' + str(img_name) + '_r_' + str(r) + '.hdr', "a+")
            hdr = """ENVI
            samples = {width}
            lines   = {height}
            bands   = 1
            header offset = 0
            file type = ENVI Standard
            data type = 1
            interleave = bsq""".format(width=width, height=height)
            f.write(hdr)
            f.close()  # close file
        _result_img.append(imageResult)
        # update sizes
        width = int(width / 2)
        height = int(height / 2)

    return _result_img


def export_results(resultQuadTree, testQuadTree, experiment_name,
                   confusionMat=None,
                   prodAccuracy=None,
                   averageAccuracy=None,
                   kappaCoeff=None,
                   title=''):
    """ Export classification results given images of result and ground-truth
    """
    H = testQuadTree[0].getHeight()
    W = testQuadTree[0].getWidth()
    C = testQuadTree[0].getNumOfClasses()
    R = len(testQuadTree)
    confusionMatrixList = []
    cohenKappaScoreList = []
    producerAccuracyList = []
    labels = np.arange(C) + 1  # [1, 2, ... , C]

    output_path = 'output/' + str(experiment_name) + '/result.txt'

    # get accuracies and confusion matrices
    accuracy = []  # list of accuracy for each resolution
    countSuccess = 0
    countTestPixel = 0
    width = W
    height = H

    for r in range(R):
        # label choosed is argmax on xs
        for h in range(height):
            for w in range(width):
                # do computation for getting accuracy
                if testQuadTree[r].pixel[0][h][w] != 0:
                    countTestPixel += 1
                    if testQuadTree[r].pixel[0][h][w] == resultQuadTree[r].pixel[0][h][w]:
                        countSuccess += 1
        accuracy.append(countSuccess / countTestPixel)
        # reset sizes
        height = int(height / 2)
        width = int(width / 2)
        # reset counters
        countSuccess = 0
        countTestPixel = 0

        y_true = testQuadTree[r].pixel.ravel()
        y_pred = resultQuadTree[r].pixel.ravel()
        if confusionMat is not None:
            confusionMatrixList.append(confusion_matrix(y_true, y_pred, labels))
        if kappaCoeff is not None:
            cohenKappaScoreList.append(cohen_kappa_score(y_true, y_pred, labels))

    if prodAccuracy is not None:
        # compute producers accuracies
        for r in range(R):
            singleResProducerAccuracies = []

            for c in range(C):
                # print(confusionMatrixList[r][c][c])
                singleResProducerAccuracies.append(confusionMatrixList[r][c][c])

            for c1 in range(C):
                count = 0
                for c2 in range(C):
                    count += confusionMatrixList[r][c1][c2]
                    # count += confusionMatrixList[r][c2][c1] #for user accuracies
                singleResProducerAccuracies[c1] /= count

            producerAccuracyList.append(singleResProducerAccuracies)

    if averageAccuracy is not None:
        averageAccuracy = []
        for r in range(R):
            _sum = 0
            for c in range(C):
                _sum += producerAccuracyList[r][c]
            _sum /= C
            averageAccuracy.append(_sum)

    # write accuracies in file
    f = open(output_path, "a+")
    f.write(title + '\n\r')
    f.write('overall accuracy\n\r')
    for r in reversed(range(R)):
        # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
        f.write('r = ' + str(r) + ' -> ' + str(accuracy[r]) + '\n')
    f.write('\n')
    # close file
    f.close()

    if prodAccuracy is not None:
        # write producer accuracies in file
        f = open(output_path, "a+")
        f.write('producer accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + '\n')
            for c in range(C):
                # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
                f.write('c = ' + str(c + 1) + ' -> ' + str(producerAccuracyList[r][c]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if averageAccuracy is not None:
        # write average accuracies in file
        f = open(output_path, "a+")
        f.write('average accuracies\n\r')
        for r in reversed(range(R)):
            f.write('r = ' + str(r) + ' -> ' + str(averageAccuracy[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if kappaCoeff is not None:
        # write cohen kappa score in file
        f = open(output_path, "a+")
        f.write('cohen kappa scores\n\r')
        for r in reversed(range(R)):
            # print('overall accuracy in r = ', r, ' -> ', accuracy[r])
            f.write('r = ' + str(r) + ' -> ' + str(cohenKappaScoreList[r]) + '\n')
        f.write('\n')
        # close file
        f.close()

    if confusionMat is not None:
        # write confusion matrix in file
        f = open(output_path, "a+")
        f.write('confusion matrices\n\r')
        for r in reversed(range(R)):
            mat = np.matrix(confusionMatrixList[r])
            # with open('outfile.txt','wb') as f:
            for line in mat:
                np.savetxt(f, line, fmt='%i', delimiter='    ')
            f.write('\n')
        # close file
        f.close()

    # add a blank line at the bottom
    f = open(output_path, "a+")
    f.write('\n')
    f.close()


def create_partial_post_folder(experiment, dataset, _method):
    add_str = ''  # additional string to allow different dataset's versions
    if experiment["dataset"]["small"]:
        add_str = '/small'
    partial_folder = 'datasets/' + dataset["name"] + '/' + add_str + '/partial_post'
    _file = partial_folder + '/' + str(_method) + '_bott-up.npy'
    return partial_folder, _file


def note_down_computation_time(output_path, start_time):
    # write computation time in result file
    f = open(output_path + '/result.txt', "a+")
    f.write('computing time: ' + str(time.time() - start_time) + ' sec' + '\n\r\n\r')
    f.close()  # close file