import pandas as pd
import numpy as np
import argparse
import copy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVC
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
import compute_performance

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', default="svm", type=str, nargs='?', help='classifier')
parser.add_argument('--optimize', default="acc", type=str, nargs='?', help='measure to optimize')
parser.add_argument('--method', default="ccral", type=str, nargs='?', help='method')
args = parser.parse_args()

np.random.seed(123)

classifier = args.classifier
measure_opt = args.optimize
if measure_opt == "acc":
    measure_index = -2
if measure_opt == "auc":
    measure_index = -1
method = args.method # ccral, cf

datasets = ['german', 'compas', 'twins', 'bank', 'adult']
n_run = 5

for dataset in datasets:
    print('running ', dataset)

    data_frame = pd.read_csv('data/' + dataset + '.csv')
    removed_cols = ['outcome']
    sen_feat = 'sex_male'
    if dataset == 'twins':
        sen_feat = 'treatment'
    if dataset == 'bank':
        sen_feat = 'marital_married'

    x_cols = list(data_frame.columns)
    x_cols = [e for e in x_cols if not e in removed_cols]
    index_frame = pd.read_csv('data/' + dataset + '_data_split.csv')

    for run in range(n_run):
        fold = run + 1
        print('running fold ', fold)
        df = index_frame[index_frame['fold'] == fold]
        train_indices = list(df['train_indices'])[0]
        train_indices = list(map(int, train_indices.split(' ')))
        valid_indices = list(df['valid_indices'])[0]
        valid_indices = list(map(int, valid_indices.split(' ')))
        test_indices = list(df['test_indices'])[0]
        test_indices = list(map(int, test_indices.split(' ')))

        train_df = data_frame[data_frame.index.isin(train_indices)]
        x_train = train_df[x_cols].values
        y_train = train_df['outcome'].values
        valid_df = data_frame[data_frame.index.isin(valid_indices)]
        x_valid = valid_df[x_cols].values
        y_valid = valid_df['outcome'].values
        test_df = data_frame[data_frame.index.isin(test_indices)]
        x_test = test_df[x_cols].values
        y_test = test_df['outcome'].values

        # update train_indices
        train_indices = range(len(x_train))
        # find index of sensitive feature
        sen_feat_idx = np.where(train_df.columns.values == sen_feat)[0][0]

        # predict step 1
        classifier_initialization = 8
        if classifier == "svm":
            clf = LinearSVC(random_state=classifier_initialization)
            clf.fit(x_train, y_train)
            # get probability of class 1
            y_train_pred = clf._predict_proba_lr(x_train)[:, 1]
            y_valid_pred = clf._predict_proba_lr(x_valid)[:, 1]
            y_test_pred = clf._predict_proba_lr(x_test)[:, 1]

        best_clf = clf
        best_perf = compute_performance.perf(y_valid, y_valid_pred)[measure_index]
        print('performance of initial classifier on valid set:', best_perf)
        test_perf = compute_performance.perf(y_test, y_test_pred)[measure_index]
        print('performance of initial classifier on test set:', test_perf)

        if method == "cf":
            alphas = [0.5]
        if method == "ccral":
            no_iterations = 10
            alphas = np.linspace(0, 0.5, no_iterations)

        # generate counterfactual samples for all training samples
        # get value of treatment feature in each training sample
        treatment_train = x_train[:, sen_feat_idx]
        treatment_opposite_train = 1 - treatment_train
        x_train_counterfactual = copy.deepcopy(x_train)
        x_train_counterfactual[:, sen_feat_idx] = treatment_opposite_train
        # compute distances between each counterfactual sample and all other samples
        dist_counterfactual_train = euclidean_distances(x_train_counterfactual, x_train)
        # obtain matched samples
        # sort distances and get the second closet as the first one is itself
        matched_indices = [np.argsort(dist_counterfactual_train[idx, :])[0] for idx in train_indices]
        y_train_counterfactual = y_train[matched_indices]

        for alpha in alphas:
            # find uncertain samples which are difficult to predict
            difficult_indices = [idx for idx in train_indices if (0.5 - alpha) <= y_train_pred[idx] <= (0.5 + alpha)]
            if len(difficult_indices) > 0:
                print("no of uncertain samples: {}".format(len(difficult_indices)))
                # obtain counterfactual samples of uncertain samples
                x_difficult_counterfactual = x_train_counterfactual[difficult_indices]
                y_difficult_counterfactual = y_train_counterfactual[difficult_indices]

                # create training data with real samples
                x_train_valid = np.append(x_train, x_valid, axis=0)
                y_train_valid = np.append(y_train, y_valid, axis=0)
                # create training data with real and counterfactual samples
                x_train_new = np.append(x_train_valid, x_difficult_counterfactual, axis=0)
                y_train_new = np.append(y_train_valid, y_difficult_counterfactual, axis=0)

                # predict step 2
                if classifier == "svm":
                    current_clf = LinearSVC(random_state=classifier_initialization)
                    current_clf.fit(x_train_new, y_train_new)
                    # get probability of class 1
                    y_valid_pred = current_clf._predict_proba_lr(x_valid)[:, 1]

                current_perf = compute_performance.perf(y_valid, y_valid_pred)[measure_index]

                if current_perf > best_perf:
                    best_perf = current_perf
                    best_clf = current_clf

                print('len(y_train): {}, alpha: {}, current_perf: {}, best_perf: {}'.
                      format(len(y_train), round(alpha, 3), current_perf, best_perf))

        if classifier == "svm":
            y_test_pred = best_clf._predict_proba_lr(x_test)[:, 1]
        ret = compute_performance.perf(y_test, y_test_pred)
        print("performance of best classifier on test set:")
        print(ret, '\n')

        # save results to file
        with open('result/{}_{}_{}_opt_{}_run{}.csv'.format(method, classifier, dataset, measure_opt, fold), 'w') as f:
            f.write('Fold,F1,Precision,Recall,Specificity,Accuracy,AUC\n')
            f.write(str(fold) + ',' + ','.join(map(str, ret)) + '\n')


