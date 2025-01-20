from image import Image
import json


def initialize_experiment(name):
    """It retrieves the information to run the experiment and to deal with the dataset.
    The experiment asd the dataset are described in the json files placed respective in the
    experiments and the datasets folders.
    """
    # get experiment specification
    with open('experiments/' + str(name) + '.json') as f:
        experiment = json.load(f)
    # get dataset information specification
    with open('datasets/' + str(experiment["dataset"]["num"]) + '.json') as f:
        dataset = json.load(f)

    return experiment, dataset


def get_variables(experiment, dataset):
    """
    It retrieves a tuple of variables that are used to run correctly the experiment.
    :return: R -> num of resolutions
    :return: W -> width of the biggest image
    :return: H -> height of the biggest image
    :return: C -> num od classes
    :return: _method -> method to estimate the statistics
    :return: statistics -> boolen, tells if to estimate the statistics
    :return: data_fus -> boolen, tells if to do the data fusion
    """
    data_used = get_small_dataset_str(experiment)

    R = dataset["R"]
    W = dataset[data_used]["W"]
    H = dataset[data_used]["H"]
    C = dataset[data_used]["C"]
    _method = experiment["statistic_estimation"]["method"]
    statistics = experiment["functionality"]["class_statistics"]
    data_fus = experiment["functionality"]["data_fusion"]

    return R, W, H, C, _method, statistics, data_fus


def load_quad_tree(imgQuadTree, mapQuadTree, testQuadTree, experiment, dataset):
    R, W, H, C, _method, _, _ = get_variables(experiment, dataset)
    data_used = get_small_dataset_str(experiment)

    width = W
    height = H
    for r in range(0, R):
        # create images
        img = Image(width, height, dataset["B"][r])
        trainMap = Image(width, height)
        testMap = Image(width, height)
        # load data
        img.load('datasets/' + dataset["name"] + '/' + str(dataset[data_used]["image_path"][r]),
                 dataset["byte_sample_image"][r])
        trainMap.load('datasets/' + dataset["name"] + '/' + str(dataset[data_used]["train_path"][r]),
                      dataset["byte_sample_map"])
        trainMap.stdMaker()
        testMap.load('datasets/' + dataset["name"] + '/' + str(dataset[data_used]["test_path"][r]),
                     dataset["byte_sample_map"])
        testMap.stdMaker()

        # equilibrate classes statistics
        if experiment["dataset"]["equilib_classes"] == True:
            trainMap.equilibrateClasses(experiment["dataset"]["factor_equilb"])

        # removes not natives bands
        if dataset["selective_bands"] == True:
            img.cutBand(dataset["band_to_be_cutted"][r])

        # insert data in quad tree
        imgQuadTree.append(img)
        mapQuadTree.append(trainMap)
        testQuadTree.append(testMap)

        # reduce sizes
        width = int(width / 2)
        height = int(height / 2)


def get_small_dataset_str(experiment):
    """It returns a string to treat the small dataset version, if any.
    """
    if experiment["dataset"]["small"]:
        data_used = 'small_data'
    else:
        data_used = 'data'
    return data_used
