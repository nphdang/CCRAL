import glob
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', default="svm", type=str, nargs='?', help='classifier')
parser.add_argument('--optimize', default="acc", type=str, nargs='?', help='measure to optimize')
args = parser.parse_args()

classifier = args.classifier
measure_opt = args.optimize
if measure_opt == "acc":
    measure_name = "Accuracy"
if measure_opt == "auc":
    measure_name = "AUC"

datasets = ['german', 'bank', 'twins', 'compas', 'adult']
methods = ['Standard', 'Counterfactual', 'CCRAL']
NUM_FOLD = 5

if "Standard" in methods:
    # join standard classifier results
    all_files = []
    for dataset in datasets:
        files = glob.glob('result/std_{}_{}*.csv'.format(classifier, dataset))
        all_files = np.append(all_files, files)
    # remove "1.csv" in file names
    groups = set([e[:-5] for e in all_files])
    for group in groups:
        subs = [e for e in all_files if e.startswith(group)]
        subs.sort()
        ls = [pd.read_csv(e) for e in subs]
        if len(ls) == NUM_FOLD:
            df = pd.concat(ls)
            df.to_csv(group[:-4] + '_ALL.csv', index=None)

if "Counterfactual" in methods:
    # join counterfactual results
    all_files = []
    for dataset in datasets:
        files = glob.glob('result/cf_{}_{}*.csv'.format(classifier, dataset))
        all_files = np.append(all_files, files)
    # remove "1.csv" in file names
    groups = set([e[:-5] for e in all_files])
    for group in groups:
        subs = [e for e in all_files if e.startswith(group)]
        subs.sort()
        ls = [pd.read_csv(e) for e in subs]
        if len(ls) == NUM_FOLD:
            df = pd.concat(ls)
            df.to_csv(group[:-4] + '_ALL.csv', index=None)

if "CCRAL" in methods:
    # join ccral results
    all_files = []
    for dataset in datasets:
        files = glob.glob('result/ccral_{}_{}*.csv'.format(classifier, dataset))
        all_files = np.append(all_files, files)
    # remove "1.csv" in file names
    groups = set([e[:-5] for e in all_files])
    for group in groups:
        subs = [e for e in all_files if e.startswith(group)]
        subs.sort()
        ls = [pd.read_csv(e) for e in subs]
        if len(ls) == NUM_FOLD:
            df = pd.concat(ls)
            df.to_csv(group[:-4] + '_ALL.csv', index=None)

# summarize performance of each method
with open('result/_result_{}_{}.csv'.format(classifier, measure_opt), 'w') as f:
    f.write('Dataset,Method,{}\n'.format(measure_name))
    for dataset in datasets:
        for method in methods:
            if method == "Standard":
                result = pd.read_csv('result/std_{}_{}_ALL.csv'.format(classifier, dataset))
                result = result.mean()*100
            if method == "Counterfactual":
                result = pd.read_csv('result/cf_{}_{}_opt_{}_ALL.csv'.format(classifier, dataset, measure_opt))
                result = result.mean() * 100
            if method == "CCRAL":
                result = pd.read_csv('result/ccral_{}_{}_opt_{}_ALL.csv'.format(classifier, dataset, measure_opt))
                result = result.mean()*100
            performance = round(result[measure_name], 2)
            ls = [dataset, method, performance]
            st = ','.join(map(str, ls)) + '\n'
            f.write(st)

df = pd.read_csv('result/_result_{}_{}.csv'.format(classifier, measure_opt))
print(df)


