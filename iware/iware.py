# implementation of iWare-E for PAWS
# Lily Xu
# May 2019

import sys
import time
import pickle

import pandas as pd
import numpy as np

from scipy.optimize import minimize

from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn import tree
from sklearn.svm import LinearSVC, SVC
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.gaussian_process.kernels import RBF

from itertools import product

from gpc import GaussianProcessClassifier

NUM_COLS_TO_SKIP = 6    # number of extraneous columns in 'x' features CSV file

POSITIVE_LABEL = 1      # how a positive label is encoded in the data
RANDOM_SEED = None        # could be None
N_JOBS = 1 # -1 to use max

# parameters for bagging classifier
NUM_ESTIMATORS = 32 #32 #50
MAX_SAMPLES = 0.8
MAX_FEATURES = .5

# verbose output if == 1
VERBOSE = 0





###########################################################
# modify GPR to serve as a classifier
# and offer variance results
###########################################################

def gpr_predict_proba(self, x, return_var=False):
    mean_r, std = self.predict(x, return_std=True)
    prob = 1 / (1 + np.exp(mean_r - 0.5))
    prob = prob.reshape(-1, 1)

    # form array with predictions for both classes
    predictions = np.concatenate((prob, 1 - prob), axis=1)

    if return_var:
        var = [x**2 for x in std]
        return predictions, var
    else:
        return predictions

# def gpr_get_var(self, x):
#     _, std = self.predict(x, return_std=True)
#
#     return [x**2 for x in std]


GaussianProcessRegressor.predict_proba = gpr_predict_proba
# GaussianProcessRegressor.get_var = gpr_get_var

def rf_predict_proba(self, x, return_var=False, train_x=None):
    predictions = self.predict_proba_orig(x)

    import forestci as fci

    if return_var:
        assert train_x is not None

        var = fci.random_forest_error(self, train_x, x)
        return predictions, var
    else:
        return predictions

RandomForestClassifier.predict_proba_orig = RandomForestClassifier.predict_proba
RandomForestClassifier.predict_proba = rf_predict_proba


###########################################################
# utility functions
###########################################################

# given training and testing sets, normalize data to zero mean, unit variance
def normalize_data(train, test):
    scaler = StandardScaler()
    # fit only on training data
    scaler.fit(train)
    # apply normalization to training and test data
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test

# by maximizing F1 score?
def determine_threshold(label, predict_test_pos_probs, num_thresholds=50):
    # TODO: previously, used tpr-(1-fpr)
    # fpr, tpr, thresholds = metrics.roc_curve(label, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
    # or maybe scaled, like 2*tpr - (1-fpr)?

    thresholds = np.linspace(0, 1, num_thresholds)
    f1         = np.zeros(thresholds.size)
    precision  = np.zeros(thresholds.size)
    recall     = np.zeros(thresholds.size)
    auprc      = np.zeros(thresholds.size)

    for i in range(num_thresholds):
        predict_labels = predict_test_pos_probs > thresholds[i]
        predict_labels = predict_labels.astype(int)

        f1[i]        = metrics.f1_score(label, predict_labels)
        precision[i] = metrics.precision_score(label, predict_labels, pos_label=POSITIVE_LABEL)
        recall[i]    = metrics.recall_score(label, predict_labels, pos_label=POSITIVE_LABEL)

        precision_vals, recall_vals, _ = metrics.precision_recall_curve(label, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
        auprc[i]     = metrics.auc(recall_vals, precision_vals)

        if VERBOSE:
            print('threshold: {:.4f} | f1: {:.4f},  precision: {:.4f}, recall: {:.4f}, AUPRC: {:.4f}'.format(thresholds[i], f1[i], precision[i], recall[i], auprc[i]))

    # opt = np.argmax(f1)
    opt = np.argmax(auprc)
    print('optimal threshold {:.4f}, with f1 {:.4f}, precision {:.4f}, recall {:.4f}, AUPRC {:.4f}'.format(thresholds[opt], f1[opt], precision[opt], recall[opt], auprc[opt]))

    return thresholds[opt]


# evaluate the ML model on the test set by print all relevant metrics for the test set
def evaluate_results(test_y, predict_test_pos_probs):
    output = []

    # compute optimal threshold and determine labels
    opt_threshold = determine_threshold(test_y, predict_test_pos_probs)
    predict_test = (predict_test_pos_probs > opt_threshold).astype(int)

    predict_test_neg_probs = np.ones(predict_test_pos_probs.shape) - predict_test_pos_probs
    predict_test_probs = np.concatenate((predict_test_neg_probs.reshape(1,-1), predict_test_pos_probs.reshape(1,-1)), axis=0).transpose()

    # select the prediction column with probability of assigned label
    predict_test_label_probs = predict_test_probs[[i for i in range(predict_test.shape[0])], tuple(predict_test)]

    fpr, tpr, _ = metrics.roc_curve(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
    output.append('AUC: {:.5f}'.format(metrics.auc(fpr, tpr)))

    precision_vals, recall_vals, _ = metrics.precision_recall_curve(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)
    output.append('AUPRC: {:.5f}'.format(metrics.auc(recall_vals, precision_vals)))  # area under precision-recall curve
    #output.append('average precision score: {:.5f}'.format(metrics.average_precision_score(test_y, predict_test_pos_probs, pos_label=POSITIVE_LABEL)))

    output.append('precision: {:.5f}'.format(metrics.precision_score(test_y, predict_test, pos_label=POSITIVE_LABEL)))
    recall = metrics.recall_score(test_y, predict_test, pos_label=POSITIVE_LABEL)
    output.append('recall: {:.5f}'.format(recall))
    output.append('F1 score: {:.5f}'.format(metrics.f1_score(test_y, predict_test, pos_label=POSITIVE_LABEL)))

    percent_positive = np.where(predict_test == POSITIVE_LABEL)[0].shape[0] / predict_test.shape[0]
    l_and_l = recall ** 2 / percent_positive
    max_ll = 1 / (test_y.sum() / test_y.shape[0])
    output.append('L&L %: {:.5f} ({:.5f} / {:.5f})'.format(100 * (l_and_l / max_ll), l_and_l, max_ll))

    output.append('cross entropy: {:.5f}'.format(metrics.log_loss(test_y, predict_test_probs)))

    output.append('average prediction probability: {:.5f}'.format(predict_test_label_probs.mean()))
    output.append('accuracy: {:.5f}'.format(metrics.accuracy_score(test_y, predict_test)))
    output.append('cohen\'s kappa: {:.5f}'.format(metrics.cohen_kappa_score(test_y, predict_test)))  # measures inter-annotator agreement
    output.append('F-beta score: {:.5f}'.format(metrics.fbeta_score(test_y, predict_test, 2, pos_label=POSITIVE_LABEL))) # commonly .5, 1, or 2. if 1, then same as f1

    return '\n'.join(output)


###########################################################
# setup data
###########################################################

def setup_data(x_filename, y_filename):
    # read in features
    features_raw = pd.read_csv(x_filename)
    features_raw.drop(columns=features_raw.columns[0], inplace=True)

    patrol_effort = features_raw['current_patrol_effort'].values
    section_col   = features_raw['section'].values
    year_col      = features_raw['year'].values

    # features_raw.drop(columns=['temp', 'precip'], inplace=True)

    # don't use current_patrol_effort as a feature
    features_raw.drop(columns='current_patrol_effort', inplace=True)

    # read in labels
    labels_raw = pd.read_csv(y_filename)
    labels_raw.drop(columns=labels_raw.columns[0], inplace=True)


    features = features_raw.values[:, NUM_COLS_TO_SKIP:]
    labels   = labels_raw.values[:, NUM_COLS_TO_SKIP]


    feature_names = list(features_raw.columns)[NUM_COLS_TO_SKIP:]
    print('feature names {}'.format(feature_names))

    return features_raw, features, feature_names, labels, patrol_effort, section_col, year_col



###########################################################
# iWare-E class
###########################################################

class iWare:
    def __init__(self, method, num_classifiers, park, year):
        self.method = method
        self.num_classifiers = num_classifiers
        self.park = park
        self.year = year
        self.patrol_thresholds = None
        self.classifiers = None
        self.weights = None         # weights for classifiers
        self.train_x_norm = None  # normalized numpy array of train_x

    # get classifier used as base estimator in bagging classifier
    def get_base_estimator(self):
        if self.method == 'gp':
            # kernel = ConstantKernel(1e-20, (1e-25, 1e-15))* RBF(length_scale=1)
            kernel = 1.0 * RBF(length_scale=1.0)
            #kernel = 1.0 * RBF(length_scale=20.0)
            # look at Matern kernel?

            # ********
            # Aaron suggests printing out length scale

            #kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-05, 1e5))
            # optimizer=None to keep kernel parameters in place
            # n_restarts_optimizer=5,
            base_estimator = GaussianProcessClassifier(kernel=kernel, random_state=RANDOM_SEED, warm_start=True, max_iter_predict=100, n_jobs=-1)

            # base_estimator = GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_SEED, n_restarts_optimizer=0, normalize_y=True)
        elif self.method == 'svm':
            base_estimator = SVC(gamma='auto', random_state=RANDOM_SEED)
        elif self.method == 'linear-svc':
            base_estimator = LinearSVC(max_iter=5000, random_state=RANDOM_SEED)
        elif self.method == 'dt':
            base_estimator = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
        else:
            raise Exception('method \'{}\' not recognized'.format(self.method))

        return base_estimator



    # get overall classifier to use
    def get_classifier(self, use_balanced):
        if self.method == 'rf':
            return RandomForestClassifier(n_estimators=NUM_ESTIMATORS,
                criterion='gini', max_depth=None, min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                max_features=MAX_FEATURES, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                bootstrap=True, oob_score=False, n_jobs=N_JOBS,
                random_state=RANDOM_SEED, verbose=VERBOSE,
                warm_start=False, class_weight=None)
            # return RandomForestRegressor(n_estimators=NUM_ESTIMATORS,
            #     criterion='mse', max_depth=None, min_samples_split=2,
            #     min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            #     max_features=MAX_FEATURES, max_leaf_nodes=None,
            #     min_impurity_decrease=0.0, min_impurity_split=None,
            #     bootstrap=True, oob_score=False,
            #     n_jobs=N_JOBS, random_state=RANDOM_SEED,
            #     verbose=VERBOSE, warm_start=False)

        base_estimator = self.get_base_estimator()

        # GPs don't need a bagging classifier
        # return base_estimator
        if self.method == 'gp':
            return base_estimator

        # balanced bagging classifier used for datasets with strong label imbalance
        # (e.g. SWS in Cambodia)
        elif use_balanced:
            return BalancedBaggingClassifier(base_estimator=base_estimator,
                n_estimators=NUM_ESTIMATORS, max_samples=MAX_SAMPLES,
                max_features=MAX_FEATURES,
                bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False,
                sampling_strategy='majority', #sampling_strategy=0.8,
                replacement=True, n_jobs=N_JOBS,
                random_state=RANDOM_SEED, verbose=VERBOSE)

        # non-balanced bagging classifier used for other datasets
        else:
            return BaggingClassifier(base_estimator=base_estimator,
                n_estimators=NUM_ESTIMATORS, max_samples=MAX_SAMPLES,
                max_features=MAX_FEATURES,
                bootstrap=True, bootstrap_features=False,
                oob_score=False, warm_start=False, n_jobs=N_JOBS,
                random_state=RANDOM_SEED, verbose=VERBOSE)




    ###########################################################
    # classification
    ###########################################################

    def get_patrol_thresholds(self, train_effort):
        patrol_threshold_percentile = np.linspace(0, 100, self.num_classifiers, endpoint=False)
        patrol_thresholds = np.percentile(train_effort, patrol_threshold_percentile)
        print('percentiles {}'.format(patrol_threshold_percentile))
        print('patrol thresholds {}'.format(patrol_thresholds))
        return patrol_thresholds


    def get_vote_matrix(self):
        vote_power = np.identity(self.num_classifiers)                           # identity matrix
        # vote_power = np.tril(np.ones((self.num_classifiers, self.num_classifiers)))   # lower triangle
        # vote_power = np.triu(np.ones((self.num_classifiers, self.num_classifiers)))   # upper triangle

        # build triangular vote qualification matrix
        # vote_qual = np.triu(np.ones((self.num_classifiers, self.num_classifiers)))
        vote_qual = np.ones((self.num_classifiers, self.num_classifiers))

        # create combined vote matrix
        vote_combine = np.multiply(vote_power, vote_qual)

        # normalize column-wise
        vote_combine = vote_combine / vote_combine.sum(1)[:,None]

        return vote_combine


    # train a set of classifiers using provided data
    def train_classifiers(self, train_x, train_y, train_effort, use_balanced):
        classifiers = []
        for i in range(self.num_classifiers):
            #### filter data
            # get training data for this threshold
            idx = np.where(np.logical_or(train_effort >= self.patrol_thresholds[i], train_y == POSITIVE_LABEL))[0]

            if i > 0 and self.patrol_thresholds[i] == self.patrol_thresholds[i-1]:
                print('threshold {} same as previous, value {}. skipping'.format(i, self.patrol_thresholds[i]))
                classifiers.append(None)
                continue

            if idx.size == 0:
                print('no training points found for classifier {}, threshold = {}'.format(i, self.patrol_thresholds[i]))
                classifiers.append(None)
                continue

            train_x_filter = train_x[idx, :]
            train_y_filter = train_y[idx]

            print('filtered data: {}. num positive labels {}'.format(train_x_filter.shape, np.sum(train_y_filter)))

            if np.sum(train_y_filter) == 0:
                print('no positive labels in this subset of the training data. skipping classifier {}'.format(i))
                classifiers.append(None)
                continue

            # select and train a classifier
            classifier = self.get_classifier(use_balanced)

            print('classifier {}, threshold {}, num x {}'.format(i, self.patrol_thresholds[i], train_x_filter.shape))
            start_time = time.time()

            # fit training data
            classifier.fit(train_x_filter, train_y_filter)

            print('  train time: {:.2f} seconds, score: {:.5f}'.format(
                    time.time() - start_time,
                    classifier.score(train_x_filter, train_y_filter)))

            classifiers.append(classifier)

            print('-------------------------------------------')

        return classifiers


    # given a set of trained classifiers, compute optimal weights
    def get_classifier_weights(self, classifiers, reserve_x, reserve_y):
        # test classifiers on all data points
        predictions = []
        for i in range(self.num_classifiers):
            if classifiers[i] is None:
                predictions.append(np.zeros(reserve_y.shape))
                continue

            curr_predictions = classifiers[i].predict(reserve_x)
            predictions.append(curr_predictions)
        predictions = np.array(predictions).transpose()

        # define loss function
        def evaluate_ensemble(weights):
            # ensure we don't get NaN values
            if np.isnan(weights).any():
                return 1e9

            weighted_predictions = np.multiply(predictions, weights)
            weighted_predictions = np.sum(weighted_predictions, axis=1)

            score = metrics.log_loss(reserve_y, weighted_predictions)

            return score
            # evaluate score
            # auprc = metrics.average_precision_score(reserve_y, weighted_predictions, pos_label=POSITIVE_LABEL)
            #
            # # pass in negative to minimize
            # return -auprc

        # constrain weights to sum to 1
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        # bound weights to be between 0 and 1
        bounds = [(0,1)] * self.num_classifiers

        # random restarts with random initial weights
        #best_weights = np.ones(self.num_classifiers) / self.num_classifiers # default: equal weights
        best_weights = None
        best_score = 1e9

        # ensure we have enough positive samples
        unique_vals, unique_counts = np.unique(reserve_y, return_counts=True)
        unique_dict = dict(zip(unique_vals, unique_counts))
        if VERBOSE:
            print(unique_dict)
        if 1 not in unique_dict or unique_dict[1] < 5:
            print('  not enough positive labels. skipping')
            return best_weights

        for _ in range(10):
            w = np.random.rand(self.num_classifiers)
            w = w / np.sum(w)

            res = minimize(evaluate_ensemble, w, method='SLSQP', bounds=bounds, constraints=cons)
            if res.fun < best_score:
                best_weights = res.x
                best_score = res.fun

        if VERBOSE:
            print('best score {}, weights {}'.format(best_score, np.around(best_weights, 3)))
        return best_weights


    def train_iware(self, all_train_x, all_train_y, all_train_effort, use_balanced=False, nsplits=5):
        self.patrol_thresholds = self.get_patrol_thresholds(all_train_effort)

        print('shape x', all_train_x.shape)
        print('shape y', all_train_y.shape)
        print('shape train_effort', all_train_effort.shape)

        # print('k-fold cross validation, k = {}'.format(nsplits))

        # skf = StratifiedKFold(nsplits, shuffle=True)
        #
        # all_weights = np.zeros((nsplits, self.num_classifiers, self.num_classifiers))
        #
        #
        #
        # # reserve some test data as validation set
        # # to assign weights to classifiers
        # k = 0
        # for train_index, reserve_index in skf.split(all_train_x, all_train_y):
        #     train_x = all_train_x[train_index]
        #     train_y = all_train_y[train_index]
        #     train_effort = all_train_effort[train_index]
        #
        #     reserve_x = all_train_x[reserve_index]
        #     reserve_y = all_train_y[reserve_index]
        #     reserve_effort = all_train_effort[reserve_index]
        #
        #
        #     print('-------------------------------------------')
        #     print('training classifiers with limited train data, k = {}'.format(k))
        #     print('-------------------------------------------')
        #
        #     classifiers = self.train_classifiers(train_x, train_y, train_effort, use_balanced)
        #
        #
        #     print('-------------------------------------------')
        #     print('finding weights for classifiers')
        #     print('-------------------------------------------')
        #
        #     # ----------------------------------------------
        #     # find appropriate weights for classifiers
        #     # ----------------------------------------------
        #     for i in range(self.num_classifiers):
        #         #### filter data
        #         # find points within specified threshold
        #         if i == 0:
        #             idx = np.where(reserve_effort < self.patrol_thresholds[i+1])[0]
        #         elif i == self.num_classifiers - 1:
        #             idx = np.where(self.patrol_thresholds[i] <= reserve_effort)[0]
        #         else:
        #             idx = np.where(np.logical_and(self.patrol_thresholds[i] <= reserve_effort, reserve_effort < self.patrol_thresholds[i+1]))[0]
        #
        #         filter_x = reserve_x[idx]
        #         filter_y = reserve_y[idx]
        #         print('classifier {}: {} points, {} positive'.format(i, filter_x.shape[0], np.count_nonzero(filter_y == POSITIVE_LABEL)))
        #         weights = self.get_classifier_weights(classifiers, filter_x, filter_y)
        #
        #         # if weights were not set, assign classifier weight to just 1 at classifier location (corresponding to the matrix diagonal)
        #         if weights is None:
        #             weights = np.zeros(self.num_classifiers)
        #             weights[i] = 1
        #
        #         all_weights[k, i, :] = weights
        #
        #     k += 1
        #
        # # average all classifier weights
        # self.weights = all_weights.mean(0)
        # print('weights: ', np.around(self.weights, 4))

        # self.weights = np.eye(self.num_classifiers)
        self.weights = self.get_vote_matrix()

        print('-------------------------------------------')
        print('training classifiers with all train data')
        print('-------------------------------------------')

        self.classifiers = self.train_classifiers(all_train_x, all_train_y, all_train_effort, use_balanced)

        # TODO: does this need to be moved?
        # need train_x later for random forest variance
        if self.method == 'rf':
            self.train_x_norm = np.copy(all_train_x)


    def test_iware(self, test_x, test_y, test_effort, output_path):
        if self.patrol_thresholds is None:
            raise ValueError('No patrol thresholds. test_iware() may not have been called.')
        if self.classifiers is None:
            raise ValueError('No classifiers. test_iware() may not have been called.')
        if self.weights is None:
            raise ValueError('No weights. test_iware() may not have been called.')

        for i in range(len(self.weights)):
            print('classifier {}, weights {}, sum {}'.format(i, np.around(self.weights[i], 4), self.weights[i].sum()))

        # # test classifiers on all data points
        # predictions = []
        # for i in range(self.num_classifiers):
        #     if self.classifiers[i] is None:
        #         # predictions.append(None)
        #         predictions.append(np.zeros(test_x.shape))
        #         continue
        #
        #     curr_predictions = self.classifiers[i].predict(test_x)
        #     predictions.append(curr_predictions)
        # predictions = np.array(predictions).transpose()
        #
        # weighted_predictions = np.multiply(predictions, self.weights)
        # weighted_predictions = np.sum(weighted_predictions, axis=1)
        #
        # evaluate_results(test_y, weighted_predictions)
        #
        # return

        ###########


        # predicted probability of illegal activity observation on each data point
        num_test = test_y.shape[0]
        weighted_predictions = np.zeros(num_test)
        weighted_variances = np.zeros(num_test)

        all_predictions = np.zeros((num_test, self.num_classifiers))

        if self.method == 'gp' or self.method == 'rf':
            all_variances = np.zeros((num_test, self.num_classifiers))

        # TODO: can i do this portion more efficiently?
        # compute the classification interval for each point
        classification = np.zeros(num_test)
        for i in range(num_test):
            smaller_thresholds = np.where(test_effort[i] > self.patrol_thresholds)[0]
            # patrol effort at this point may be less than all threshold values
            if len(smaller_thresholds) == 0:
                classification[i] = 0
            else:
                classification[i] = smaller_thresholds[-1]
        classification = classification.astype(int)

        for i in range(self.num_classifiers):
            if self.classifiers[i] is None:
                print('classifier {} is none; skipping'.format(i))
                continue

            start_time = time.time()

            # compute variance
            if self.method == 'gp' or self.method == 'rf':
                if self.method == 'rf':
                    assert self.train_x_norm is not None
                    curr_predictions, curr_variances = self.classifiers[i].predict_proba(test_x, return_var=True, train_x=self.train_x_norm)
                elif self.method == 'gp':
                    # curr_predictions, curr_variances = self.classifiers[i].predict_proba(test_x, return_var=True)
                    curr_predictions = self.classifiers[i].predict_proba(test_x)
                    curr_variances = self.classifiers[i].predict_var(test_x)
                    # curr_variances = curr_variances[:, 1]

                # normalize variance values
                curr_variances = curr_variances - np.min(curr_variances)
                curr_variances = curr_variances / np.max(curr_variances)

                all_variances[:, i] = curr_variances

            # this method doesn't allow variance :(
            else:
                curr_predictions = self.classifiers[i].predict_proba(test_x)

            curr_predictions = curr_predictions[:, 1]   # probability of positive label
            all_predictions[:, i] = curr_predictions

            # TODO: make more efficient!
            multiplier = np.zeros(num_test)
            for j in range(num_test):
                multiplier[j] = self.weights[classification[j]][i]

            # scale increase by the multiplier for each data point
            weighted_predictions += np.multiply(curr_predictions, multiplier)

            if self.method == 'gp' or self.method == 'rf':
                weighted_variances += np.multiply(curr_variances, multiplier)

            print(' classifier {}, test time {:.3f}'.format(i, time.time() - start_time))


        # save out predictions to CSV
        print('  save out predictions...')
        predictions_df = pd.DataFrame(data=all_predictions, columns=['threshold={}'.format(thresh) for thresh in self.patrol_thresholds])
        predictions_out = '{}/{}_{}_predictions.csv'.format(output_path, self.method, self.num_classifiers)
        print('  {}'.format(predictions_out))
        predictions_df.to_csv(predictions_out)

        # save out variances to CSV
        if self.method == 'gp' or self.method == 'rf':
            print('  save out variances...')
            variances_df = pd.DataFrame(data=all_variances, columns=['threshold={}'.format(thresh) for thresh in self.patrol_thresholds])
            variances_df.to_csv('{}/{}_{}_variances.csv'.format(output_path, self.method, self.num_classifiers))

            combined_df = pd.DataFrame({'predictions': weighted_predictions, 'variances': weighted_variances})
            combined_df.to_csv('{}/{}_{}_weighted_pred_var.csv'.format(output_path, self.method, self.num_classifiers))



        ### evaluate
        results = evaluate_results(test_y, weighted_predictions)
        print(results)

        f = open('{}/{}_{}.txt'.format(output_path, self.method, self.num_classifiers), 'w')
        f.write('park {}, test year {}\n'.format(self.park, self.year))
        f.write('method {}, num_classifiers {}\n'.format(self.method, self.num_classifiers))
        f.write('thresholds {}\n'.format(self.patrol_thresholds))
        f.write('\n\n')
        f.write(results)
        f.write('\n\n\n')
        f.write('weights\n{}\n'.format(np.around(self.weights, 5)))
        f.close()

        pickle_data = {'park': self.park,
                        'num_classifiers': self.num_classifiers, 'method': self.method,
                        #'classifiers': self.classifiers,
                        'weights': self.weights,
                        'thresholds': self.patrol_thresholds,
                        'predictions': weighted_predictions,
                        'results': results
                        }
        pickle.dump(pickle_data, open('{}/{}_{}.p'.format(output_path, self.method, self.num_classifiers), 'wb'))

        # # display performance on only first classifier
        # # only using the first is the same as no iWare-E ensembling
        # print('-------------------------------------------')
        # print('testing - only first classifier')
        # print('-------------------------------------------')
        #
        # predict_test_probs = classifiers[0].predict_proba(test_x)
        # predict_test_pos_probs = predict_test_probs[:,1]
        # evaluate_results(test_y, predict_test_pos_probs)
        #
        # # write out predictions to file
        # predict_out = pd.DataFrame(data={'predictions': predict_test_pos_probs, 'labels': test_y})
        # predict_out.to_csv('output/test_predictions_{}_{}_method_{}.csv'.format(self.park, TEST_YEAR, self.method))



    ###########################################################
    # run train/test code to evaluate predictive models
    ###########################################################

    # prepare data: split into train/test and normalize
    def train_test_split_by_year(self, features, labels, patrol_effort, year_col, test_year, test_section=None, section_col=None):
        # specifying the section is optional
        if test_section is not None:
            assert section_col is not None

        if test_section:
            # just one section of test data
            train_idx = np.where(np.logical_or(year_col < test_year, section_col < test_section))[0]
            test_idx  = np.where(np.logical_and(year_col == test_year, section_col == test_section))[0]
        else:
            # full year of test data
            train_idx = np.where(year_col < test_year)[0]
            test_idx  = np.where(year_col == test_year)[0]

        train_x      = features[train_idx, :]
        train_y      = labels[train_idx]
        train_effort = patrol_effort[train_idx]

        test_x      = features[test_idx, :]
        test_y      = labels[test_idx]
        test_effort = patrol_effort[test_idx]

        train_x, test_x = normalize_data(train_x, test_x)

        print('train x, y', train_x.shape, train_y.shape)
        print('test x, y ', test_x.shape, test_y.shape)
        print('patrol effort train, test ', train_effort.shape, test_effort.shape)

        return train_x, test_x, train_y, test_y, train_effort, test_effort


    ###########################################################
    # iWare-E for predicting future risk
    ###########################################################

    # use all provided data to make predictions
    def make_predictions(self, predict_section, features_raw, features, feature_names,
            labels, patrol_effort, section_col, input_static_feats, output_path,
            test_temp=None, test_precip=None, gpp_filename=None):
        print('time to make some predictions!')

        predict_year = self.year

        # ----------------------------------------------
        # get training data
        # ----------------------------------------------
        # use all data before specified (predict_year, predict_section)
        train_idx = np.where(np.logical_or(features_raw['year'] < predict_year,
            np.logical_and(features_raw['year'] == predict_year, features_raw['section'] < predict_section)))[0]

        print('  features shape', features_raw.shape)
        print('  train_dx ', len(train_idx), train_idx)


        train_x = features[train_idx, :]
        train_y = labels[train_idx]
        train_patrol_effort = patrol_effort[train_idx]

        # ----------------------------------------------
        # get data to predict on
        # ----------------------------------------------
        if predict_section == 0:
            prev_year = predict_year - 1
            num_section = np.max(section_col)
            print('  num section', num_section)
            prev_section = num_section
        else:
            prev_year = predict_year
            prev_section = predict_section - 1

        print('  test section: year {}, section {}'.format(predict_year, predict_section))
        print('  prev section: year {}, section {}'.format(prev_year, prev_section))


        # ----------------------------------------------
        # set up data arrays
        # ----------------------------------------------
        # get past patrol effort for the test section
        prev_section_idx = np.where(np.logical_and(features_raw['year'] == prev_year, features_raw['section'] == prev_section))
        past_patrol_effort = patrol_effort[prev_section_idx]

        prev_section_spatial_id = features_raw['spatial_id'].values[prev_section_idx]
        patrol_effort_df = pd.DataFrame({'spatial_id': prev_section_spatial_id,
                                         'past_patrol_effort': past_patrol_effort})

        # get all static features
        static_features = pd.read_csv(input_static_feats)
        static_features.drop(columns=static_features.columns[0], inplace=True)

        # create features array and add in past_patrol_effort
        predict_x_df = static_features.join(patrol_effort_df.set_index('spatial_id'), on='spatial_id', how='left')
        predict_x_df['past_patrol_effort'].fillna(0, inplace=True)

        # add climate info
        if test_temp is not None and test_precip is not None:
            predict_x_df['temp']   = test_temp * np.ones(static_features.shape[0])
            predict_x_df['precip'] = test_precip * np.ones(static_features.shape[0])

        # add GPP info
        if gpp_filename is not None:
            new_gpp = pd.read_csv('../preprocess_consolidate/belum_traponly_combined/1000/output/all_3month/GPP_2019_0.csv')
            predict_x_df['gpp'] = new_gpp['2019-0']

        # arrange columns to match training data
        store_columns = predict_x_df[['spatial_id', 'x', 'y']]
        predict_x_df.drop(columns=['spatial_id', 'x', 'y'], inplace=True)
        predict_x_df = predict_x_df[feature_names]
        predict_x = predict_x_df.values

        # normalize data
        train_x, predict_x = normalize_data(train_x, predict_x)


        # ----------------------------------------------
        # train classifiers
        # ----------------------------------------------
        print('training classifiers on {} points...'.format(train_x.shape))

        train_start_time = time.time()
        classifiers = self.train_iware(train_x, train_y, train_patrol_effort)
        total_train_time = time.time() - train_start_time
        print('total train time {:.3f}'.format(total_train_time))


        # ----------------------------------------------
        # run classifiers to get set of predictions
        # ----------------------------------------------

        # intiialize array to store predictions from each classifier
        print('making predictions on year {} section {}... {} points'.format(predict_year, predict_section, predict_x.shape))
        final_predictions = np.zeros((predict_x.shape[0], self.num_classifiers))

        if self.method == 'gp' or self.method == 'rf':
            final_variances = np.zeros((predict_x.shape[0], self.num_classifiers))

        # make predictions with each classifier
        for i in range(self.num_classifiers):
            start_time = time.time()

            # this classifier had no training points, so we skip it
            if self.classifiers[i] is None:
                final_predictions[:, i] = np.zeros((final_predictions.shape[0]))
                continue

            if self.method == 'gp' or self.method == 'rf':
                if self.method == 'rf':
                    curr_predictions, curr_variances = self.classifiers[i].predict_proba(predict_x, return_var=True, train_x=train_x)
                else:
                    # curr_predictions, curr_variances = self.classifiers[i].predict_proba(predict_x, return_var=True)
                    curr_predictions = self.classifiers[i].predict_proba(predict_x)
                    curr_variances = self.classifiers[i].predict_var(predict_x)

                # curr_variances = curr_variances[:, 1]

                print('variance min {} max {}'.format(np.min(curr_variances), np.max(curr_variances)))

                # normalize variance values
                # curr_variances = curr_variances - np.min(curr_variances)
                # curr_variances = curr_variances / np.max(curr_variances)

                final_variances[:, i] = curr_variances

            else:
                curr_predictions = self.classifiers[i].predict_proba(predict_x)

            curr_predictions = curr_predictions[:, 1]   # probability of positive label

            final_predictions[:, i] = curr_predictions


        # predict_x_df.to_csv('predict_x.csv', encoding='utf-16')
        # np.savetxt('predict_x_norm.csv', predict_x, delimiter=',', encoding='utf-16', fmt='%.3f')
        # np.savetxt('train_x.csv', self.train_x_norm, delimiter=',', encoding='utf-16', fmt='%.3e')
        # np.savetxt('train_x_float.csv', self.train_x_norm, delimiter=',', encoding='utf-16', fmt='%.3f')

        # save out predictions to CSV
        print('  save out predictions...')
        predictions_df = pd.DataFrame(data=final_predictions, columns=['threshold={}'.format(thresh) for thresh in self.patrol_thresholds])
        predictions_df = pd.concat([store_columns, predictions_df], axis=1)
        predictions_filename = '{}/predictions_{}_{}_method_{}_{}.csv'.format(output_path, self.park, predict_year, self.method, self.num_classifiers)
        print('  {}'.format(predictions_filename))
        predictions_df.to_csv(predictions_filename)

        # save out variances to CSV
        if self.method == 'gp' or self.method == 'rf':
            print('  save out variances...')
            variances_df = pd.DataFrame(data=final_variances, columns=['threshold={}'.format(thresh) for thresh in self.patrol_thresholds])
            variances_df = pd.concat([store_columns, variances_df], axis=1)
            variances_df.to_csv('{}/variances_{}_{}_method_{}_{}.csv'.format(output_path, self.park, predict_year, self.method, self.num_classifiers))

        return predictions_df



    # used for post-hoc analysis of field test data
    # (we want to ignore the true data and pretend we don't know it)
    def field_test_make_predictions(self, predict_year, predict_section, features, labels, patrol_effort, input_static_feats,
            feature_names):
        print('time to make some predictions!')

        # ----------------------------------------------
        # GET TRAINING DATA
        # ----------------------------------------------
        # get last quarter of patrol effort
        predict_mask = np.logical_and(features_raw['year'] == predict_year, features_raw['section'] == predict_section)

        predict_train_idx = np.where(np.logical_not(predict_mask))[0]
        train_x = features[predict_train_idx, :]
        train_patrol_effort = patrol_effort[predict_train_idx]
        train_y = labels[predict_train_idx]

        # ----------------------------------------------
        # GET DATA FOR PREDICTIONS
        # ----------------------------------------------
        # get indices from available cells at the specified time interval
        predict_idx = np.where(predict_mask)[0]

        # get past patrol effort for those available cells
        predict_spatial_id = features_raw['spatial_id'].values[predict_idx]
        predict_patrol_effort = patrol_effort[predict_idx]
        patrol_effort_df = pd.DataFrame({'spatial_id': predict_spatial_id, 'past_patrol_effort': predict_patrol_effort})

        # get all static features
        static_features = pd.read_csv(input_static_feats)

        # create features array
        predict_x_df = static_features.join(patrol_effort_df.set_index('spatial_id'), on='Var1', how='left')
        predict_x_df['past_patrol_effort'].fillna(0, inplace=True)

        predict_x_df.drop(columns=['Var1', 'x', 'y'], inplace=True)

        # arrange columns to match training data
        predict_x_df = predict_x_df[feature_names]
        predict_x = predict_x_df.values


        # ----------------------------------------------
        # normalize data
        # ----------------------------------------------
        train_x, predict_x = normalize_data(train_x, predict_x)


        # ----------------------------------------------
        # train classifiers
        # ----------------------------------------------
        # print('training classifiers on {} points...'.format(predict_train_x.shape))

        train_start_time = time.time()
        classifiers = self.train_iware(predict_train_x, train_y, train_patrol_effort)
        total_train_time = time.time() - train_start_time
        print('total train time {:.3f}'.format(total_train_time))


        # ----------------------------------------------
        # run classifiers to get set of predictions
        # ----------------------------------------------

        # intiialize array to store predictions from each classifier
        print('making predictions on year {} section {}... {} points'.format(predict_year, predict_section, predict_x.shape))
        final_predictions = np.zeros((predict_x.shape[0], self.num_classifiers))

        # make predictions with each classifier
        for i in range(self.num_classifiers):
            start_time = time.time()

            # this classifier had no training points, so we skip it
            if classifiers[i] is None:
                final_predictions[:, i] = np.zeros((final_predictions.shape[0]))
                continue

            curr_predictions = classifiers[i].predict_proba(predict_x)
            curr_predictions = curr_predictions[:, 1]   # probability of positive label

            final_predictions[:, i] = curr_predictions

        # save out predictions to CSV
        print('save out predictions...')
        predictions_df = pd.DataFrame(data=final_predictions, columns=['threshold={}'.format(thresh) for thresh in patrol_thresholds])
        # start indexing from 1 to be consistent with other files
        predictions_df.index = np.arange(1, len(predictions_df) + 1)
        predictions_df.to_csv('predictions_{}_{}_method_{}.csv'.format(self.park, predict_year, self.method)) #float_format='%.4f',



###########################################################
# variation attempts for filtering data
###########################################################

        #### filter data
        # get training data for this threshold

        # # MY MODIFIED APPROACH
        # # makes things run faster, and sometimes get decent results
        # # only points within threshold interval
        # if i == 0:
        #     idx = np.where(train_effort < patrol_thresholds[i+1])[0]
        # elif i == num_classifiers - 1:
        #     idx = np.where(train_effort >= patrol_thresholds[i])[0]
        # else:
        #     idx = np.where(np.logical_and(train_effort >= patrol_thresholds[i], train_effort < patrol_thresholds[i+1]))[0]

        # # points within threshold interval AND all positive points
        # if i == 0:
        #     idx = np.where(np.logical_or(train_effort < patrol_thresholds[i+1], train_y == POSITIVE_LABEL))[0]
        # elif i == num_classifiers - 1:
        #     idx = np.where(np.logical_or(train_effort >= patrol_thresholds[i], train_y == POSITIVE_LABEL))[0]
        # else:
        #     idx = np.where(np.logical_or(np.logical_and(train_effort >= patrol_thresholds[i], train_effort < patrol_thresholds[i+1]), train_y == POSITIVE_LABEL))[0]

        # ------------------------------------------------------------------
        # this is the original iWare-E approach
        # all points above threshold
        # if PARK == 'SWS':
        #     # don't keep positive labels for SWS because of the strong label imbalance
        #     idx = np.where(train_effort >= patrol_thresholds[i])[0]
        # else:
        #     # AND POINTS WHERE LABEL IS POSITIVE
        #     idx = np.where(np.logical_or(train_effort >= patrol_thresholds[i], train_y == POSITIVE_LABEL))[0]

###########################################################
# calibration curves
###########################################################
# from calibration_curves import *
# run_all_calibration_curves(train_x, train_y, test_x, test_y)
# sys.exit(0)
