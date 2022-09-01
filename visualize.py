import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
sns.set_context("notebook", font_scale=1.8)
plt.style.use('fivethirtyeight')

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', default="svm", type=str, nargs='?', help='classifier')
args = parser.parse_args()

classifier = args.classifier
# plot accuracy
acc_result = './result/_result_{}_acc.csv'.format(classifier)
df = pd.read_csv(acc_result, header=0, sep=",")
print("plot accuracy")
g = sns.catplot(x="Dataset", y="Accuracy", hue="Method", data=df, kind="bar", ci="sd", height=5, aspect=2, palette="Set1")
g.set_xlabels("Dataset")
g.set_ylabels("Accuracy")
for idx, p in enumerate(g.ax.patches):
    height = round(p.get_height(), 2)
    g.ax.text(p.get_x()+p.get_width()/2, height+1, str(round(height, 2)), ha="center", fontsize=10)
plt.savefig("./result/_plot_{}_accuracy.pdf".format(classifier), bbox_inches="tight")
plt.close()

# plot AUC
auc_result = './result/_result_{}_auc.csv'.format(classifier)
df = pd.read_csv(auc_result, header=0, sep=",")
print("plot AUC")
g = sns.catplot(x="Dataset", y="AUC", hue="Method", data=df, kind="bar", ci="sd", height=5, aspect=2, palette="Set1")
g.set_xlabels("Dataset")
g.set_ylabels("AUC")
for idx, p in enumerate(g.ax.patches):
    height = round(p.get_height(), 2)
    g.ax.text(p.get_x()+p.get_width()/2, height+1, str(round(height, 2)), ha="center", fontsize=10)
plt.savefig("./result/_plot_{}_auc.pdf".format(classifier), bbox_inches="tight")
plt.close()

