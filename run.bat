python standard_classifier.py --classifier svm

python ccral.py --classifier svm --optimize acc --method ccral
python ccral.py --classifier svm --optimize auc --method ccral

python ccral.py --classifier svm --optimize acc --method cf
python ccral.py --classifier svm --optimize auc --method cf

python comparison.py --classifier svm --optimize acc
python comparison.py --classifier svm --optimize auc

python visualize.py --classifier svm