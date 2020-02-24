import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn import tree
from sklearn import svm
import pickle
import os
import util
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors

from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

os.chdir('.')

EPSILON = 2.2e-16       # used to avoid division by zero errors that occur with bad data


def Find_Optimal_Cutoff(target, predicted):
    '''
    Find the optimal probability cutoff point for a classification model related to event rate
    *Parameters:
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    *Returns: list type, with optimal cutoff value
    '''
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr))
    precision, recall, threshold_precision_recall = metrics.precision_recall_curve(target, predicted)
    threshold_precision_recall = np.concatenate([threshold_precision_recall, [1]])
    i_pre = np.arange(len(precision))
    beta = 0.5 # larger than 1 => in favor of recall; smaller than 1 => in favor of precision
    f1_scores = (1 + beta * beta) * (precision * recall) / (beta * beta * precision + recall)
    # print('f1 scores:', f1_scores)
    f1_scores[np.isnan(f1_scores)] = 0
    values = pd.DataFrame(
        {
            'tf': pd.Series(tpr + (1 - fpr), index=i),
            'threshold': pd.Series(threshold, index=i),
        })
    f1_values = pd.DataFrame(
        {
            'precision': pd.Series(precision, index=i_pre),
            'recall': pd.Series(recall, index=i_pre),
            'f1': pd.Series(f1_scores, index=i_pre),
            'threshold': pd.Series(threshold_precision_recall, index=i_pre)
        })
    # sorted_indices = values.iloc[(values.tf - 0).argsort()[::-1]] # TODO
    sorted_indices = f1_values.iloc[(f1_values.f1 - 0).argsort()[::-1]]
    # print('f1 scores again: ', sorted_indices['f1'].values)
    # print 'roc  ', list(sorted_indices['threshold'])
    # print 'threshold ', list(sorted_indices['threshold_precision_recall'])
    return list(sorted_indices['threshold'])
    # return list(sorted_indices['threshold_precision_recall'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bagging Cross Validation Blackbox function')
    parser.add_argument('-r', '--resolution', default=1000, help='Input the resolution scale')
    parser.add_argument('-p', '--park', help='Input the park name', required=True)
    parser.add_argument('-c', '--category', default='All', help='Input the category')
    parser.add_argument('-m', '--method', help='Input the training method')
    parser.add_argument('-static', '--static', default=False, help='Predicting on all the static data or not (used for planning)')
    parser.add_argument('-simple', '--simple', default=False, help='Not using ensemble method; i.e. only one threshold: 0')
    parser.add_argument('-cutoff', '--cutoff', default=0, help='Input the cutoff threshold of patrol effort')

    args = parser.parse_args()

    resolution = int(args.resolution)
    park = args.park
    category = args.category
    method = args.method
    all_static = args.static
    cutoff_threshold = float(args.cutoff)
    simple_classification = args.simple
    patrol_option = 0 # PEcase 0: current+past, 1: past, 2: current, 3: none

    directory = './{0}_datasets/resolution/{1}m/input'.format(park, str(resolution))
    if simple_classification:
        output_directory = './{0}_datasets/resolution/{1}m/{2}/simple_output_{3}'.format(park, str(resolution), method, cutoff_threshold)
    else:
        output_directory = './{0}_datasets/resolution/{1}m/{2}/output_{3}'.format(park, str(resolution), method, cutoff_threshold)

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(output_directory + '/Prob/Final'):
        os.makedirs(output_directory + '/Prob/Final')
    if not os.path.exists(output_directory + '/Prob/Merged'):
        os.makedirs(output_directory + '/Prob/Merged')


    if all_static:
        test_year, test_quarter = util.test_year_quarter_by_park(park)

    if simple_classification:
        selectedThresholds = [0]
    else:
        selectedThresholds = util.selected_threshold_by_park(park)

    print('selected thresholds', selectedThresholds)
    currentPatrolEffortList = util.selected_finer_threshold_by_park(park)
    pastPatrolEffortList = util.selected_threshold_by_park(park)

    evaluationResults_original = pd.DataFrame(index=['Threshold','bestCutOff', 'AUC','Precision', 'Recall', 'F1',
                                             'L&L', 'max_L&L', 'L&L %',
                                             'number of positive in training', 'number of all training',
                                             'number of positive in validation', 'number of all validation',
                                             'number of positive in testing', 'number of all testing', 'effective positive prediction'],
                                     columns=[np.arange(0, len(selectedThresholds))])
    evaluationResults = evaluationResults_original.copy(deep=True)
    evaluationResults_constraint = evaluationResults_original.copy(deep=True)

    X_all = pd.read_csv(directory + '/' + '{0}_X.csv'.format(category))
    Y_all = pd.read_csv(directory + '/' + '{0}_Y.csv'.format(category))

    X_all_copy = X_all.copy()
    #X_all.drop(['ID_Global', 'Year', 'Quarter', 'ID_Spatial'], inplace=True, axis=1)
    X_all.drop(['ID_Global', 'Year', 'Quarter', 'ID_Spatial', 'x', 'y'], inplace=True, axis=1)
    X_all = X_all.values

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(X_all[:,1:])
    X_all[:,1:] = scaler.transform(X_all[:,1:])

    dataPointIndicator_all = Y_all[['ID_Global', 'Year', 'Quarter', 'ID_Spatial', 'x', 'y']]
    Y_all.drop(['ID_Global', 'Year', 'Quarter', 'ID_Spatial', 'x', 'y'], inplace=True, axis=1)
    Y_all = Y_all.values
    Y_all = Y_all.ravel()

    # ==================== separating testing set =======================
    testing_year, testing_quarter = util.test_year_quarter_by_park(park)
    cutoff_useful_threshold = cutoff_threshold
    if testing_quarter is not None:
        print('testing on year {0}, quarter {1}...'.format(testing_year, testing_quarter))
        testing_indices = (dataPointIndicator_all['Year'] == testing_year) & (dataPointIndicator_all['Quarter'] == testing_quarter)
        training_indices = -testing_indices

        testing_indices = testing_indices & (X_all[:,0] >= cutoff_useful_threshold) # removing the testing data below the threshold
    else:
        print('testing on year {0}...'.format(testing_year))
        testing_indices = (dataPointIndicator_all['Year'] == testing_year) & (X_all[:,0] >= cutoff_useful_threshold)
        training_indices = -testing_indices

        testing_indices = testing_indices & (X_all[:,0] >= cutoff_useful_threshold) # removing the testing data below the threshold


    X_all_test = X_all[testing_indices]
    Y_all_test = Y_all[testing_indices]
    X_all_train = X_all[training_indices]
    Y_all_train = Y_all[training_indices]
    dataPointIndicator_test_all = dataPointIndicator_all[testing_indices]

    print('X_all_test:', X_all_test.shape)

    No_of_Split = 5
    No_of_Iteration = 1
    # ===================== shuffle training set ========================
    X_all_train, Y_all_train = sklearn.utils.shuffle(X_all_train, Y_all_train)

    # ======================== blackbox parsing =========================
    indicatorList = ['ID_Spatial', 'x', 'y']

    if all_static:
        X_all_static = pd.read_csv(directory + '/' + 'allStaticFeat.csv')
        X_all_static = X_all_static.rename(columns={'Var1': 'ID_Spatial'})
        X_all_static['currentPatrolEffort'] = 0.0
        X_all_static['pastPatrolEffort'] = 0.0
        column_names = X_all_static.columns.tolist()
        X_all_static = X_all_static[column_names[:3] + column_names[-2:] + column_names[3:-2]]

        ID_spatial_2_patrol_effort = {}
        for index, row in X_all_copy.iterrows():
            ID_spatial = row['ID_Spatial']
            patrol_effort = row['currentPatrolEffort']
            year = row['Year']
            quarter = row['Quarter']
            if year == test_year and quarter == test_quarter:
                ID_spatial_2_patrol_effort[ID_spatial] = patrol_effort

        for index, row in X_all_static.iterrows():
            ID_spatial = row['ID_Spatial']
            if ID_spatial_2_patrol_effort.has_key(ID_spatial):
                # print('ID spatial', ID_spatial)
                X_all_static['pastPatrolEffort'][index] = ID_spatial_2_patrol_effort[ID_spatial]
            else:
                X_all_static['pastPatrolEffort'][index] = 0

        # ================ remove current patrol and past patrol =============
        if patrol_option == 1:
            X_all_static.drop(['currentPatrolEffort'], inplace=True, axis=1)
        elif patrol_option == 2:
            X_all_static.drop(['pastPatrolEffort'], inplace=True, axis=1)
        elif patrol_option == 3:
            X_all_static.drop(['currentPatrolEffort', 'pastPatrolEffort'], inplace=True, axis=1)

        Y_static_prob_predict_merged_list = []
        Y_static_prob_predict_merged_attack_list = []
        Y_static_prob_predict_merged_average = X_all_static.copy()[['ID_Spatial', 'x', 'y']]
        Y_static_prob_predict_merged_attack_average = X_all_static.copy()[['ID_Spatial', 'x', 'y']]
        for i in range(No_of_Split * No_of_Iteration):
            Y_static_prob_predict_merged_list.append(X_all_static.copy()[['ID_Spatial', 'x', 'y']])
            Y_static_prob_predict_merged_attack_list.append(X_all_static.copy()[['ID_Spatial', 'x', 'y']])


    # ======================== cross-validating ==========================
    count = 0

    skf = StratifiedKFold(n_splits=No_of_Split)

    for train_index, validate_index in skf.split(X_all_train, Y_all_train):
        X_train_main, X_validate_main = X_all_train[train_index], X_all_train[validate_index]
        Y_train_main, Y_validate_main = Y_all_train[train_index], Y_all_train[validate_index]
        train_current_patrol = X_train_main[:,0]
        validate_current_patrol = X_validate_main[:,0]
        test_current_patrol = X_all_test[:,0]

        if patrol_option == 0:
            X_test_main = X_all_test
            Y_test_main = Y_all_test
        elif patrol_option == 1:
            X_train_main = X_train_main[:,1:]
            X_validate_main  = X_validate_main[:,1:]
            X_test_main = X_all_test[:,1:]
            Y_test_main = Y_all_test
        elif patrol_option == 2:
            X_train_main = np.concatenate([X_train_main[:,0], X_train_main[:,2:]], axis=1)
            X_validate_main  = np.concatenate([X_validate_main[:,0],  X_validate_main[:,2:]],  axis=1)
            X_test_main  = np.concatenate([X_all_test[:,0],  X_all_test[:,2:]],  axis=1)
            Y_test_main = Y_all_test
        elif patrol_option == 3:
            X_train_main = X_train_main[:,2:]
            X_validate_main  = X_validate_main[:,2:]
            X_test_main  = X_all_test[:,2:]
            Y_test_main = Y_all_test

        # ===================================================

        #print (validate_index)
        #print (len(validate_index))

        dataPointIndicator_main = dataPointIndicator_all.iloc[validate_index]


        count += 1

        for jj in range(No_of_Iteration):
            tmp_positive_sum = 0
            print ('Dataset Number: ', count, ' Round of the iteration', jj)
            colNumber = 0

            for thrsh_number in range(len(selectedThresholds)):
                thrsh = selectedThresholds[thrsh_number]
                next_thrsh = 100 if thrsh_number == len(selectedThresholds)-1 else selectedThresholds[thrsh_number+1]
                if thrsh == 0:
                    A_train = train_current_patrol >= thrsh
                    B_train = Y_train_main == 1
                    C_train = train_current_patrol < next_thrsh
                    # D_train = A_train * C_train + B_train # TODO
                    D_train = A_train + B_train # TODO
                    A_validate = validate_current_patrol >= thrsh
                    C_validate = validate_current_patrol < next_thrsh
                    if simple_classification:
                        D_validate = A_validate # & C_validate # TODO
                    else:
                        D_validate = A_validate # & C_validate # TODO

                    A_test = test_current_patrol >= thrsh
                    C_test = test_current_patrol < next_thrsh
                    D_test = A_test # & C_test # TODO

                    X_train = X_train_main[D_train]
                    Y_train = Y_train_main[D_train]

                    X_validate = X_validate_main[D_validate]
                    Y_validate = Y_validate_main[D_validate]

                    X_test = X_test_main[D_test]
                    Y_test = Y_test_main[D_test]

                    dataPointIndicator_validate = dataPointIndicator_main.iloc[D_validate]
                    dataPointIndicator_test = dataPointIndicator_test_all.iloc[D_test]

                else:
                    A_train = train_current_patrol >= thrsh
                    B_train = Y_train_main[:] == 1
                    C_train = train_current_patrol < next_thrsh
                    # D_train = A_train * C_train + B_train # TODO
                    D_train = A_train + B_train # TODO
                    X_train = X_train_main[D_train]
                    Y_train = Y_train_main[D_train]
                    print (thrsh)

                    A_validate = validate_current_patrol >= thrsh
                    C_validate = validate_current_patrol < next_thrsh
                    if simple_classification:
                        D_validate = A_validate # & C_validate # TODO
                    else:
                        D_validate = A_validate # & C_validate # TODO

                    X_validate = X_validate_main[D_validate]
                    Y_validate = Y_validate_main[D_validate]
                    #print ('C_validate', C_validate)

                    A_test = test_current_patrol >= thrsh # WARNING: MAKE SURE NOT TO MAKE INFORMATION LEAKAGE
                    C_test = test_current_patrol < next_thrsh
                    D_test = A_test # & C_test # TODO
                    X_test = X_test_main[D_test]
                    Y_test = Y_test_main[D_test]

                    dataPointIndicator_validate = dataPointIndicator_main.iloc[D_validate]
                    dataPointIndicator_test = dataPointIndicator_test_all.iloc[D_test]


                max_samples = 1.0 # ratio of sample size draw in the bagging
                if method == 'dt':
                    clf = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

                elif method == 'svm':
                    clf = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

                elif method == 'gp':
                    max_samples = min(1000, len(X_train))
                    clf = BaggingClassifier(base_estimator=GaussianProcessClassifier(kernel=RBF(length_scale=1.00), optimizer=None), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
    #
                elif method == 'balance-dt':
                    clf = BalancedBaggingClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

                elif method == 'balance-svm':
                    clf = BalancedBaggingClassifier(base_estimator=svm.SVC(), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

                elif method == 'balance-gp':
                    max_samples = min(1000, len(X_train))
                    clf = BalancedBaggingClassifier(base_estimator=GaussianProcessClassifier(kernel=RBF(length_scale=1.00), optimizer=None), n_estimators=50, max_samples=max_samples,
                                        max_features=1.0, bootstrap=True, bootstrap_features=True,
                                        oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

                elif method == 'dummy':
                    clf = DummyClassifier(strategy='stratified')

                clf.fit(X_train, Y_train)

                # ================== prediction and evaluation on validation set ===================
                ######## Prediction: Probabilities ########
                Y_validate_prob_predict = clf.predict_proba(X_validate)
                # if Y_validate_prob_predict.shape[1] == 1:
                #     Y_validate_prob_predict = np.concatenate((np.zeros(Y_validate_prob_predict.shape), Y_validate_prob_predict), axis=1)
                Y_train_prob_predict = clf.predict_proba(X_train)

                Y_validate_predict = clf.predict(X_validate)
                Y_train_predict = clf.predict(X_train)

                # find best cutoff point based on the validation set
                bestCutOff_mdl = Find_Optimal_Cutoff(Y_validate, Y_validate_prob_predict[:, -1])[0]
                # bestCutOff_mdl = 0.8
                Y_validate_predict = (Y_validate_prob_predict[:,1] >= bestCutOff_mdl).astype(int)

                # =========================== prediction on testing set ==============================
                Y_test_prob_predict = clf.predict_proba(X_test)
                Y_test_predict = (Y_test_prob_predict[:,1] >= bestCutOff_mdl).astype(int)
                # Y_test_predict = clf.predict(X_test)
                if method == 'dummy':
                    Y_test_predict = clf.predict(X_test)

                dataPointIndicator_test['Y_test_predict'] = Y_test_predict
                dataPointIndicator_test['Y_test_prob_predict'] = Y_test_prob_predict[:, -1]

                dataPointIndicator_test.to_csv(path_or_buf=output_directory + '/Prob/Prediction_'+ str(count) + '_Thresh_' +
                                str(thrsh) + '_round_' + str(jj) + '.csv', sep=',', float_format='%.2f', index=False)

                # =========================== evaluation on testing set ==============================
                fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_prob_predict[:, -1], pos_label=1)
                if len(fpr) == 1:
                    auc_mdl = 1
                else:
                    auc_mdl = metrics.auc(fpr, tpr)

                # statistics
                precision_mdl = metrics.precision_score(Y_test, Y_test_predict, pos_label=1)
                recall_mdl = metrics.recall_score(Y_test, Y_test_predict, pos_label=1)
                f1_mdl = metrics.f1_score(Y_test, Y_test_predict, pos_label=1)
                # avoid division by zero
                denominator = Y_test_predict.sum()
                if denominator == 0:
                    print('Y_test_predict.sum() is zero!')
                    denominator = EPSILON
                LL_mdl = (recall_mdl ** 2) * Y_test_predict.shape[0] / denominator
                max_LL = Y_test.shape[0] / Y_test.sum()
                # avoid division by zero
                if max_LL == 0:
                    max_LL = EPSILON
                evaluationResults.loc[:, colNumber] = [thrsh,bestCutOff_mdl, auc_mdl, precision_mdl, recall_mdl,
                                                       f1_mdl, LL_mdl, max_LL, LL_mdl * 100 / float(max_LL),
                                                       sum(Y_train), len(Y_train),
                                                       sum(Y_validate), len(Y_validate),
                                                       sum(Y_test), len(Y_test), sum(Y_test_predict[(X_test[:,0] >= thrsh) & (X_test[:,0] < next_thrsh)])]
                colNumber = colNumber + 1
                tmp_positive_sum += sum(Y_test_predict)
                print('column number:', colNumber)

                # ------------------------ blackbox prediction -------------------------
                if all_static:
                    assert(patrol_option == 0)
                    if simple_classification:
                        # threshold_list = np.arange(0,8,0.1)
                        threshold_list = np.arange(0,5,0.05)
                    else:
                        threshold_list = [thrsh, thrsh + 0.1, thrsh + 0.2, thrsh + 0.3, thrsh + 0.4]
                    for currentPatrolEffortVal in threshold_list:
                        X_static = X_all_static.copy(deep=True)
                        X_static['currentPatrolEffort'] = X_static['currentPatrolEffort'].values*0+currentPatrolEffortVal
                        X_static.drop(indicatorList, inplace=True, axis=1)
                        X_static = X_static.values
                        X_static[:,1:] = scaler.transform(X_static[:,1:])

                        ######## Prediction: Probabilities ########
                        Y_static_prob_predict = clf.predict_proba(X_static)
                        # Y_static_prob_predict_merged_list[(count - 1) * No_of_Iteration + jj] = pd.concat([Y_static_prob_predict_merged_list[(count - 1) * No_of_Iteration + jj], pd.DataFrame(Y_static_prob_predict[:,1], columns=['Y_test_prob_predict_{:.2f}'.format(currentPatrolEffortVal)])],axis=1)
                        Y_static_prob_predict_merged_list[(count - 1) * No_of_Iteration + jj] = pd.concat([Y_static_prob_predict_merged_list[(count - 1) * No_of_Iteration + jj], pd.DataFrame(Y_static_prob_predict[:,-1], columns=['Y_test_prob_predict_' + str(currentPatrolEffortVal)])],axis=1)

                        # for pastPatrolEffortVal in pastPatrolEffortList:
                        #     X_static = X_all_static.copy(deep=True)
                        #     X_static['currentPatrolEffort'] = X_static['currentPatrolEffort'].values*0+currentPatrolEffortVal
                        #     X_static['pastPatrolEffort'] = X_static['pastPatrolEffort'].values*0+pastPatrolEffortVal
                        #     X_static.drop(indicatorList, inplace=True, axis=1)
                        #     X_static = X_static.values
                        #     X_static[:,1:] = scaler.transform(X_static[:,1:])

                        #     ######## Prediction: Probabilities ########
                        #     Y_static_prob_predict = clf.predict_proba(X_static)
                        #     Y_static_prob_predict_merged_attack_list[(count - 1) * No_of_Iteration + jj] = pd.concat([Y_static_prob_predict_merged_attack_list[(count - 1) * No_of_Iteration + jj], pd.DataFrame(Y_static_prob_predict[:,-1], columns=['Y_test_prob_predict_currPE' + str(currentPatrolEffortVal) + '_pastPE' + str(pastPatrolEffortVal)])],axis=1)


            # ------------------------ output --------------------------------

            writer = pd.ExcelWriter(output_directory + '/result_' + str(count) + '_round_'+ str(jj) + '.xlsx', engine='xlsxwriter')
            evaluationResults.to_excel(writer, sheet_name='Sheet1', index_label='Experiment No.')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            format1 = workbook.add_format({'align': 'left'})
            format2 = workbook.add_format({'num_format': '0.00', 'align': 'center'})
            worksheet.set_column('A:A', 20, format1)
            worksheet.set_column('B:BB', 10, format2)
            writer.save()


            # -------------------- Merging the results of prediction on testing set -----------------------

            for threshold in selectedThresholds:
                if threshold == 0:
                    Y_test_all_predict_merged = pd.read_csv(output_directory + '/Prob/Prediction_'+ str(count)
                                                             + '_Thresh_' + str(threshold) + '_round_' + str(jj) + '.csv')
                    Y_test_all_predict_merged = pd.merge(dataPointIndicator_test_all, Y_test_all_predict_merged, on=['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial'], how='left')
                    # Y_test_prob_predict_merged.drop(['ID_Global','Y_test_predict'], inplace=True, axis=1)
                    # Y_test_prob_predict_merged.drop(['ID_Global_x', 'ID_Global_y', 'Y_test_predict'], inplace=True, axis=1)

                    Y_test_prob_predict_merged = Y_test_all_predict_merged[['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial', 'Y_test_prob_predict']]
                    Y_test_predict_merged = Y_test_all_predict_merged[['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial', 'Y_test_predict']]
                    Y_test_predict_merged = pd.merge(dataPointIndicator_test_all, Y_test_all_predict_merged, on=['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial'], how='left')
                    # Y_test_predict_merged.drop(['ID_Global', 'Y_test_prob_predict'], inplace=True, axis=1)
                    # Y_test_predict_merged.drop(['ID_Global_x', 'ID_Global_y', 'Y_test_prob_predict'], inplace=True, axis=1)

                else:
                    Y_test_all_predict_subset = pd.read_csv(output_directory + '/Prob/Prediction_' + str(count)
                                                             + '_Thresh_' + str(threshold) + '_round_' + str(jj) + '.csv')
                    Y_test_prob_predict_subset = Y_test_all_predict_subset[['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial', 'Y_test_prob_predict']]
                    # Y_test_prob_predict_subset.drop(['ID_Global', 'Y_test_predict'], inplace=True, axis=1)

                    Y_test_predict_subset = Y_test_all_predict_subset[['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial', 'Y_test_predict']]

                    df1 = Y_test_prob_predict_merged
                    df2 = Y_test_prob_predict_subset
                    df_temp = pd.merge(df1, df2, how='left', on=['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial'])
                    overwrite_indices = -np.isnan(df_temp['Y_test_prob_predict_y'].astype(float))  # the indices that df2 is not nan
                    df_temp['Y_test_prob_predict_x'][overwrite_indices] = df_temp['Y_test_prob_predict_y'][overwrite_indices]
                    Y_test_prob_predict_merged = df_temp.drop('Y_test_prob_predict_y', axis=1).rename(columns={'Y_test_prob_predict_x': 'Y_test_prob_predict'})

                    df3 = Y_test_predict_merged
                    df4 = Y_test_predict_subset
                    df_temp_2 = pd.merge(df3, df4, how='left', on=['ID_Global', 'Year', 'Quarter', 'x', 'y', 'ID_Spatial'])
                    overwrite_indices_2 = -np.isnan(df_temp_2['Y_test_predict_y'].astype(float))  # the indices that df4 is not nan
                    df_temp_2['Y_test_predict_x'][overwrite_indices_2] = df_temp_2['Y_test_predict_y'][overwrite_indices_2]
                    Y_test_predict_merged = df_temp_2.drop('Y_test_predict_y', axis=1).rename(columns={'Y_test_predict_x': 'Y_test_predict'})


            Y_test_prob_predict_merged.to_csv(path_or_buf=output_directory + '/Prob/Final/Y_test_Prob'+
                                                          str(count)+ '_round_' + str(jj) + '.csv', sep=',', float_format='%.2f', index_label='ID_Global')
            Y_test_predict_merged.to_csv( path_or_buf=output_directory + '/Prob/Final/Y_Predict'+
                                                      str(count) + '_round_' + str(jj) + '.csv', sep=',', float_format='%.2f', index_label='ID_Global')

            Y_test_predict = Y_test_predict_merged.loc[:, 'Y_test_predict']
            Y_test_prob_predict = Y_test_prob_predict_merged.loc[:, 'Y_test_prob_predict']
            print 'number of positive predicted: {0}'.format(tmp_positive_sum)
            print 'number of positive merged: {0}'.format(sum(Y_test_predict))
            # assert(tmp_positive_sum == sum(Y_test_predict))

            # ======================= merging evaluation ============================
            fpr, tpr, thresholds = metrics.roc_curve(Y_all_test, Y_test_prob_predict, pos_label=1)

            auc_mdl = metrics.auc(fpr, tpr)
            print('Last AUC:', auc_mdl)
            precision_mdl = metrics.precision_score(Y_all_test, Y_test_predict, pos_label=1)
            recall_mdl = metrics.recall_score(Y_all_test, Y_test_predict, pos_label=1)

            f1_mdl = metrics.f1_score(Y_all_test, Y_test_predict, pos_label=1)
            denominator = Y_test_predict.sum()
            if denominator == 0:
                denominator = EPSILON
            LL_mdl = (recall_mdl ** 2) * Y_test_predict.shape[0] / denominator
            max_LL = Y_all_test.shape[0] / Y_all_test.sum()

            if max_LL == 0:
                max_LL = EPSILON
            evaluationResults_merged = pd.DataFrame(
                data=[threshold, np.nan, auc_mdl,
                      precision_mdl, recall_mdl, f1_mdl,
                      LL_mdl, max_LL, LL_mdl * 100 / float(max_LL),
                      sum(Y_train_main), len(Y_train_main),
                      sum(Y_validate_main), len(Y_validate_main),
                      sum(Y_test_main), len(Y_test_main), sum(Y_test_predict)],
                index=evaluationResults.index,
                columns=['Merged'])
            # save merged evaluation
            writer = pd.ExcelWriter(output_directory + '/Merged'+ str(count) + '_round_' + str(jj) + '.xlsx', engine='xlsxwriter')
            pd.concat([evaluationResults, evaluationResults_merged],
                      axis=1).to_excel(writer, sheet_name='Sheet1', index_label='Experiment No.')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            format1 = workbook.add_format({'align': 'left'})
            format2 = workbook.add_format({'num_format': '0.00', 'align': 'center'})
            worksheet.set_column('A:A', 20, format1)
            worksheet.set_column('B:BB', 10, format2)
            writer.save()

    AUC_ = []
    Precision_ = []
    Recall_ = []
    F1_ = []
    LL_ = []
    Max_LL_ = []
    LL_Percent_ = []

    for ii in range(No_of_Split):
        i = ii + 1
        for jj in range(No_of_Iteration):
            A = pd.read_excel(output_directory + '/Merged' + str(i) + '_round_' + str(jj) + '.xlsx', sheet_name='Sheet1')

            AUC_.append(A['Merged'].values[2])
            Precision_.append(A['Merged'].values[3])
            Recall_.append(A['Merged'].values[4])
            F1_.append(A['Merged'].values[5])
            LL_.append(A['Merged'].values[6])
            Max_LL_.append(A['Merged'].values[7])
            LL_Percent_.append(A['Merged'].values[8])

            if all_static:
                if ii == 0 and jj == 0:
                    Y_static_prob_predict_merged_average = Y_static_prob_predict_merged_list[ii*No_of_Iteration + jj].copy()
                    Y_static_prob_predict_merged_attack_average = Y_static_prob_predict_merged_attack_list[ii*No_of_Iteration + jj].copy()
                else:
                    Y_static_prob_predict_merged_average += Y_static_prob_predict_merged_list[ii*No_of_Iteration + jj]
                    Y_static_prob_predict_merged_attack_average += Y_static_prob_predict_merged_attack_list[ii*No_of_Iteration + jj]

    if all_static:
        Y_static_prob_predict_merged_average = Y_static_prob_predict_merged_average / (No_of_Split * No_of_Iteration)
        Y_static_prob_predict_merged_attack_average = Y_static_prob_predict_merged_attack_average / (No_of_Split * No_of_Iteration)
        Y_static_prob_predict_merged_average.to_csv(path_or_buf=output_directory+'/Prob/Merged/Y_test_prob_blackBoxFunction_detect' + '.csv', sep=',', float_format='%.2f', index_label='ID_Global')
        Y_static_prob_predict_merged_attack_average.to_csv( path_or_buf=output_directory+'/Prob/Merged/Y_test_prob_blackBoxFunction_attack' + '.csv', sep=',', float_format='%.2f', index_label='ID_Global')


    AUC_new = np.array(AUC_)
    Precision_new = np.array(Precision_)
    Recall_new = np.array(Recall_)
    F1_new = np.array(F1_)
    LL_new = np.array(LL_)
    Max_LL_new = np.array(Max_LL_)
    LL_Percent_new = np.array(LL_Percent_)

    print(AUC_new.mean(), AUC_new.std())
    print(Precision_new.mean(), Precision_new.std())
    print(Recall_new.mean(), Recall_new.std())
    print(F1_new.mean(), F1_new.std())
    print(LL_new.mean(), LL_new.std())
    print(Max_LL_new.mean(), Max_LL_new.std())
    print(LL_Percent_new.mean(), LL_Percent_new.std())

    data = [[AUC_new.mean(), AUC_new.std()],[Precision_new.mean(),Precision_new.std()],
            [Recall_new.mean(), Recall_new.std()], [F1_new.mean(), F1_new.std()], [LL_new.mean(), LL_new.std()],
           [Max_LL_new.mean(),Max_LL_new.std()], [LL_Percent_new.mean(), LL_Percent_new.std()]]
    df = pd.DataFrame(data, index= ['AUC','Precision','Recall','F1', 'L&L','Max_L&L','L&L%'],columns=['Mean','Std'])

    writer = pd.ExcelWriter(output_directory+'/Final.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'align': 'left'})
    format2 = workbook.add_format({'num_format': '0.00', 'align': 'center'})
    worksheet.set_column('A:A', 20, format1)
    worksheet.set_column('B:BB', 10, format2)
    writer.save()

    f_output = open('./results_summary.csv', 'a+')
    f_output.write(', '.join([str(x) for x in [cutoff_threshold, AUC_new.mean(), Precision_new.mean(), Recall_new.mean(), F1_new.mean(), LL_new.mean(), Max_LL_new.mean(), LL_Percent_new.mean()]]) + '\n')
    f_output.close()


    # ============================== prediction result map ===============================
    if all_static:
        test_x = X_all_static['x'] / resolution
        test_y = X_all_static['y'] / resolution
        min_x = int(np.min(test_x))
        max_x = int(np.max(test_x))
        min_y = int(np.min(test_y))
        max_y = int(np.max(test_y))

        gridmap = [[0 for x in range(min_x, max_x+1)] for y in range(min_y, max_y+1)]
        tp_list = [[], []] # x, y
        fp_list = [[], []] # x, y
        fn_list = [[], []] # x, y
        tn_list = [[], []] # x, y

        for index, row in X_all_static.iterrows():
            gridmap[int(row['y']/resolution) - min_y][int(row['x']/resolution) - min_x] = 1

        gridmap = np.ma.masked_where(gridmap==1, gridmap)

        for index, row in Y_test_predict_merged.iterrows():
            if row['Y_test_predict'] == 1 and Y_all_test[index] == 1:   # true positive
                tp_list[0].append(int(row['x']/resolution))
                tp_list[1].append(int(row['y']/resolution))
            elif row['Y_test_predict'] == 1 and Y_all_test[index] == 0: # false positive
                fp_list[0].append(int(row['x']/resolution))
                fp_list[1].append(int(row['y']/resolution))
            elif row['Y_test_predict'] == 0 and Y_all_test[index] == 1: # false negative
                fn_list[0].append(int(row['x']/resolution))
                fn_list[1].append(int(row['y']/resolution))
            elif row['Y_test_predict'] == 0 and Y_all_test[index] == 0:   # true negative
                tn_list[0].append(int(row['x']/resolution))
                tn_list[1].append(int(row['y']/resolution))

        color = ['black',(0,0,0,0)]
        cmapm = colors.ListedColormap(color)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.subplots_adjust(left=0.2, wspace=0.6)

        ax1.scatter(tp_list[0], tp_list[1], marker='o', c='g', s=6)
        ax1.set_title('true positive')
        ax2.scatter(fp_list[0], fp_list[1], marker='o', c='r', s=6)
        ax2.set_title('false positive')
        ax3.scatter(fn_list[0], fn_list[1], marker='o', c='b', s=6)
        ax3.set_title('false negative')
        ax4.scatter(tn_list[0], tn_list[1], marker='o', c='y', s=6)
        ax4.set_title('true negative')
        if np.min(gridmap) == 0:
            ax1.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[min_x,max_x+1,min_y,max_y+1])
            ax2.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[min_x,max_x+1,min_y,max_y+1])
            ax3.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[min_x,max_x+1,min_y,max_y+1])
            ax4.imshow(np.flipud(gridmap), interpolation='none', cmap=cmapm, extent=[min_x,max_x+1,min_y,max_y+1])

        plt.grid(ls='solid', lw=0.1)
        plt.savefig(output_directory+'prediction_map.png')
        # plt.show()
