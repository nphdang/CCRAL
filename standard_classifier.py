import numpy as np
import pandas as pd
import argparse
from sklearn.svm import LinearSVC
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
import compute_performance

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', default="svm", type=str, nargs='?', help='classifier')
args = parser.parse_args()

np.random.seed(123)

classifier = args.classifier
method = "std"
datasets = ['german', 'compas', 'twins', 'bank', 'adult']
n_run = 5

for dataset in datasets:
    print('running ', dataset)
    data_frame = pd.read_csv('data/' + dataset + '.csv')
    removed_cols = ['outcome']
    x_cols = list(data_frame.columns)
    x_cols = [e for e in x_cols if not e in removed_cols]
    index_frame = pd.read_csv('data/' + dataset + '_data_split.csv')

    for i in range(n_run):
        fold = i + 1
        print('running fold ', fold)
        df = index_frame[index_frame['fold'] == fold]
        train_indices = list(df['train_indices'])[0]
        train_indices = list(map(int, train_indices.split(' ')))
        valid_indices = list(df['valid_indices'])[0]
        valid_indices = list(map(int, valid_indices.split(' ')))
        test_indices = list(df['test_indices'])[0]
        test_indices = list(map(int, test_indices.split(' ')))

        train_indices = train_indices + valid_indices

        train_df = data_frame[data_frame.index.isin(train_indices)]
        test_df = data_frame[data_frame.index.isin(test_indices)]

        x_train = train_df[x_cols].values
        y_train = train_df['outcome'].values
        x_test = test_df[x_cols].values
        y_test = test_df['outcome'].values

        # predict
        classifier_initialization = 8
        if classifier == "svm":
            clf = LinearSVC(random_state=classifier_initialization)
            clf.fit(x_train, y_train)
            # get probability of class 1
            y_pred = clf._predict_proba_lr(x_test)[:, 1]

        ret = compute_performance.perf(y_test, y_pred)
        print(ret, '\n')

        # save results to file
        with open('result/{}_{}_{}_run{}.csv'.format(method, classifier, dataset, fold), 'w') as f:
            f.write('Fold,F1,Precision,Recall,Specificity,Accuracy,AUC\n')
            f.write(str(fold) + ',' + ','.join(map(str, ret)) + '\n')


