"""
    This project has been implemented for the development of the master thesis:
    `Markov causal models for the classification of multiresolution and multisensor data`

    for any info: ale.montaldo@gmail.com
"""

from load_utilities import *
from export_result import *
from ensemble_estim_on_quadtree import ensemble_estim
from mpmOnQuadtree import method as mpm

# --- IMPORTANT
# in order to run the project it is necessary to run Python 3 and install
# scikit-learn version. the current version is fine
# but note that from the version 0.24 the rotation_forest.py will need to be fixed.


def main():
    # --- STARTING THE EXPERIMENT
    # experiment_name identify which experiment to run.
    # (must match the .json file in the experiments folder)
    experiment_name = 'tmp'
    # experiment and dataset are dictionaries imported from the json files with all the variables to run the program
    experiment, dataset = initialize_experiment(experiment_name)
    # R -> num of resolutions
    # W -> width of the biggest image
    # H -> height of the biggest image
    # C -> num od classes
    # _method -> method to estimate the statistics
    # statistics -> boolen, tells if to estimate the statistics
    # data_fus -> boolen, tells if to do the data fusion
    R, W, H, C, _method, statistics, data_fus = get_variables(experiment, dataset)
    # outputs are placed in a folder named experiment_name in the output folder
    output_path = set_output_location(experiment_name)

    # --- QUAD TREE LOADING
    # obs: 0 is the finest resolution, R-1 is the coarsest resolution
    imgQuadTree = []  # list of images for each resolution layer (Quadtree)
    mapQuadTree = []  # list of classification maps for each resolution layer (Quadtree)
    testQuadTree = []  # list of ground truth for each resolution layer (Quadtree)
    load_quad_tree(imgQuadTree, mapQuadTree, testQuadTree, experiment, dataset)


    # --- STATISTICS ESTIMATION with ensemble learning
    if experiment["functionality"]["class_statistics"]:
        predicted_labels, site_statistics = ensemble_estim(imgQuadTree, mapQuadTree, _method)  # out format [r][c][h][w]
        out_img = img_from_data(predicted_labels, H, W, R, output_path, _method, _from='label')
        export_results(out_img, testQuadTree, experiment_name,
                       title="Overall accuracy after " + _method + " estimation")

    # --- DATA FUSION --------------------------
    if experiment["functionality"]["data_fusion"]:
        # --- INITIALIZATION
        print('set transition probabilities')
        resolutionTranProb = mpm.computeTransProb(C, experiment["method"]["theta"])
        spatialTranProb = mpm.computeTransProb(C, experiment["method"]["theta"])
        print('compute prior')
        # prior is a list of arrays containing the prior probabilities for each layer
        prior = mpm.computePrior(mapQuadTree, R, C, resolutionTranProb)

        # ---get transition contributions (from father, from neighbourhood)
        # obs: delta_{ijkh} that appears in eq.(6) icip. with neighbour set composed by left and upper site
        print('compute transition contributions')
        transitionContrib = mpm.get_trans_contrib(R, C, resolutionTranProb, spatialTranProb, prior, num_neighbours=2)
        transitionContribSingle = mpm.get_trans_contrib(R, C, resolutionTranProb, spatialTranProb, prior,
                                                        num_neighbours=1)

        partial_folder, _file = create_partial_post_folder(experiment, dataset, _method)

        # --- BOTTOM-UP-PASS
        # do it if it has never been done or if it is forced to be recomputed
        if not os.path.isfile(_file) or experiment["functionality"]["force_bottom_up"]:
            print('BOTTOM-UP-PASS started')
            # partialPost is p(xs|y_d(s)), where y_d(s) the observations of the descendants
            partialPost = mpm.bottom_up(site_statistics, resolutionTranProb, prior, R, C, H, W, _method)

            # save partialPost in file
            # storeTensor(partialPost, 'data/output/'+str(mainName)+'/partialPost/partialPost.raw', 8)
            print('saving partialPost at root')
            # check if directory for output exist, if not creates it
            pathlib.Path(partial_folder).mkdir(parents=True, exist_ok=True)
            _tmp_file = partial_folder + '/' + str(_method) + '_bott-up'
            np.save(_tmp_file, partialPost)

        # partial posts are available
        print('loading partialPost from file')
        partialPost = np.load(_file)

        # --- TOP-DOWN-PASS
        print('TOP-DOWN-PASS started')
        partialPost = mpm.top_down(R, H, W, experiment, partialPost, transitionContrib, transitionContribSingle,
                                   partial_folder)

        img_name = 'MPM_' + str(experiment["method"]["top_down"])
        img_name += '_' + str(experiment["statistic_estimation"]["method"])
        img = img_from_data(partialPost, H, W, R, output_path, img_name)
        title = "Overall accuracy after quad-tree, " + str(experiment["method"]["top_down"]) + " version"
        export_results(img, testQuadTree, experiment_name,
                       confusionMat=True,
                       prodAccuracy=True,
                       averageAccuracy=True,
                       kappaCoeff=True,
                       title=title)

        note_down_computation_time(output_path, start_time)


if __name__ == "__main__":
    start_time = time.time()
    main()
