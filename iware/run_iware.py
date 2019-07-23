from iware import *

import sys
import argparse

plot = False

if plot:
    import matplotlib
    import matplotlib.pyplot as plt


# suppress warnings that arise from testing thresholds (e.g. precision 0)
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


###########################################################
# set parameters
###########################################################

parser = argparse.ArgumentParser(description='Evaluate iWare-E performance or make predictions')
parser.add_argument('--park', '-p', help='park name', type=str, required=True)
parser.add_argument('--year', '-y', help='test year', type=int, required=True)
parser.add_argument('--method', '-m', help='weak learner method', type=str, required=True)
parser.add_argument('--section', '-s', help='test section', type=int, required=False)
parser.add_argument('--num', '-n', help='number of classifiers for iWare-E', type=int, default=10)
parser.add_argument('--predict', help='whether to predict or evaluate (train/test)', action='store_true')
parser.add_argument('--iware-off', help='turn off iWare-E', action='store_false')

args = parser.parse_args()

PARK = args.park
TEST_YEAR = args.year
TEST_SECTION = args.section
method = args.method
num_classifiers = args.num      # 'I' parameter in iWare-E papers
PREDICT = args.predict
iWareE = args.iware_off

# to make predictions, we need a section specified to predict on
if PREDICT:
    assert TEST_SECTION is not None


# parameters for processing the data

# use only a selected number of points
CROP_POINTS = False
NUM_CROP_POINTS = 1000

# use only a selected number of features
CROP_FEATURES = False
NUM_CROP_FEATURES = 5

# input_path = '../preprocess_consolidate/belum_traponly_combined/1000/output/'


# test_temp = 0.8
# test_precip = 0.4
input_path = '../inputs/ICDE_input'
x_filename = '{}/{}_{}/All_X.csv'.format(input_path, PARK, TEST_YEAR)
y_filename = '{}/{}_{}/All_Y.csv'.format(input_path, PARK, TEST_YEAR)
static_features_filename = '{}/{}_allStaticFeat.csv'.format(input_path, PARK)

# output_path = './output/ref_{}_{}'.format(PARK, TEST_YEAR)
output_path = './output/{}_{}'.format(PARK, TEST_YEAR)

# make output folder
import os
if not os.path.exists(output_path):
    os.makedirs(output_path)

# # redirect output for all print statements to a saved text file
# sys.stdout = open('output/out_{}_{}_{}.txt'.format(PARK, TEST_YEAR, method), 'w')

print('park = {}, test year = {}, method = {}, iWareE = {}'.format(PARK, TEST_YEAR, method, iWareE))

if PARK == 'SWS':
    BALANCED_CLASSIFIER = True
else:
    BALANCED_CLASSIFIER = False


features_raw, features, feature_names, labels, patrol_effort, section_col, year_col = setup_data(x_filename, y_filename)

iware = iWare(method, num_classifiers, PARK, TEST_YEAR)

if PREDICT:
    iware.make_predictions(TEST_SECTION, features_raw, features, feature_names,
            labels, patrol_effort, section_col, static_features_filename, #test_temp, test_precip,
            output_path)

    sys.exit(0)


# train_x, test_x, train_y, test_y, train_effort, test_effort = train_test_split(features, labels, patrol_effort, test_size = .02, train_size = .1)

train_x, test_x, train_y, test_y, train_effort, test_effort = iware.train_test_split_by_year(features, labels, patrol_effort, year_col, TEST_YEAR)


###########################################################
# iWare-E
###########################################################

if iWareE:
    print('-------------------------------------------')
    print('with iWare-E, method = {}'.format(method))
    print('-------------------------------------------')

    train_start_time = time.time()
    iware.train_iware(train_x, train_y, train_effort, BALANCED_CLASSIFIER)
    total_train_time = time.time() - train_start_time
    print('total train time {:.3f}'.format(total_train_time))


    print('-------------------------------------------')
    print('testing')
    print('-------------------------------------------')

    test_start_time = time.time()
    iware.test_iware(test_x, test_y, test_effort, output_path)
    total_test_time = time.time() - test_start_time
    print('total train time {:.3f}, total test time {:.3f}, num_classifiers {}'.format(total_train_time, total_test_time, num_classifiers))

    print('park = {}, test year = {}, method = {}, iWareE = {}'.format(PARK, TEST_YEAR, method, iWareE))


else:
    print('-------------------------------------------')
    print('without iWare-E, method = {}'.format(method))
    print('-------------------------------------------')

    print('-------------------------------------------')
    print('training')
    print('-------------------------------------------')

    classifier = iware.get_classifier(BALANCED_CLASSIFIER)

    train_start_time = time.time()
    classifier.fit(train_x, train_y)
    train_predictions = classifier.predict(train_x)
    # train_predictions_probs = classifier.predict_proba(train_x)
    total_train_time = time.time() - train_start_time
    print('train time: {:.2f} seconds'.format(total_train_time))


    print('-------------------------------------------')
    print('testing')
    print('-------------------------------------------')

    test_start_time = time.time()
    predict_test_probs = classifier.predict_proba(test_x)
    total_test_time = time.time() - test_start_time

    # evaluate performance
    predict_test_pos_probs = predict_test_probs[:,1]
    results = evaluate_results(test_y, predict_test_pos_probs)
    results += '\ntotal train time {:.3f}, total test time {:.3f}'.format(total_train_time, total_test_time)
    print(results)

    f = open('{}/{}_no_iware.txt'.format(output_path, method), 'w')
    f.write('park {}, test year {}, method {}\n'.format(PARK, TEST_YEAR, method))
    f.write('\n\n')
    f.write(results)
    f.close()


sys.exit(0)

if plot:
    # plot ROC
    def plot_roc(fpr, tpr):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set(xlabel='false positive rate', ylabel='true positive rate', title='ROC')
        ax.grid()
        fig.savefig("roc.png")
        plt.show()

    # plot precision-recall curve
    def plot_precision_recall(recall_vals, precision_vals):
        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals)
        ax.set(xlabel='recall', ylabel='precision', title='precision-recall curve')
        ax.grid()
        fig.savefig("precision_recall_curve.png")
        plt.show()




###########################################################
# crop to specified number of points
###########################################################

# if CROP_POINTS:
#     # get random shuffling
#     permute = np.random.permutation(features.shape[0])
#     features = features[permute, :]
#     patrol_effort = patrol_effort[permute]
#     labels = labels[permute]
#     year = year[permute]
#
#     features = features[:NUM_CROP_POINTS, :]
#     patrol_effort = patrol_effort[:NUM_CROP_POINTS]
#     labels = labels[:NUM_CROP_POINTS]
#     year = year[:NUM_CROP_POINTS]




###########################################################
# feature importance
###########################################################

from sklearn.ensemble import ExtraTreesClassifier

def feature_importance(X, y):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=RANDOM_SEED)
    clf = clf.fit(X, y)
    return clf.feature_importances_


# print('feature importance: ', np.around(feature_importance(features, labels), decimals=3))

# select the top # most important features
if CROP_FEATURES:
    importance = feature_importance(features, labels)
    sorted = np.argsort(importance)
    print(sorted)
    print('top {} most important features: {}'.format(NUM_CROP_FEATURES, np.flip(sorted[-NUM_CROP_FEATURES:], axis=0)))

    features = features[:, sorted[-NUM_CROP_FEATURES:]]

# from sklearn.feature_selection import RFECV
# from sklearn.svm import SVR
# print('estimator...')
# estimator = SVR(kernel="linear")
# print('selector...')
# selector = RFECV(estimator, step=1, cv=3, verbose=1)
# print('fit...')
# selector = selector.fit(features, labels)
# print('done! ...')
# print('support', selector.support_)
# print('ranking', selector.ranking_)
# print('grid scores',selector.grid_scores_)

# sys.exit(0)


###########################################################
# feature selection
###########################################################
# from feature_selection import feature_selection
# base_estimator = get_base_estimator(method)
# feature_classifier = get_classifier(base_estimator, BALANCED_CLASSIFIER)
# feature_selection(train_x, train_y, test_x, test_y, classifier=feature_classifier, title='{} ({})'.format(PARK, TEST_YEAR), feature_names=feature_names)
# sys.exit(0)
