from sklearn.metrics import log_loss
import pandas as pd
import glob
import os

#Read data
folder = 'Data\Simulation\TrainingData'
link = 35.0
lane = 3.0
training_files = glob.glob(os.path.join(folder, '*.csv'))
list_files = [x for x in training_files if str(int(link))+str(int(lane)) in x]
counter = 0
data = pd.DataFrame()
for file in list_files:
    if data.empty:
        data = pd.read_csv(file, header = 0)
        counter += 1
    else:
        data_new = pd.read_csv(file,header = 0)
        data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
        data = pd.concat([data, data_new])
        counter += 1

cycles= sorted(set(data.Current_cycle.to_list()))
y = {}
for i in range(len(cycles)):
    state_now = data.loc[data.Current_cycle == cycles[i]].State.to_list()[0]
    y[cycles[i]] = state_now
from collections import Counter
counts = Counter(list(y.values()))
long = len(list(y))
ratios = [counts[0]/long, counts[1]/long, counts[2]/long, counts[3]/long,counts[4]/long]

def calculate_log_loss(class_ratio, multi=10000):
    #https://towardsdatascience.com/estimate-model-performance-with-log-loss-like-a-pro-9f47d13c8865
    if sum(class_ratio) != 1.0:
        print("warning: Sum of ratios should be 1 for best results")
        class_ratio[-1] += 1 - sum(class_ratio)  # add the residual to last class's ratio

    actuals = []
    for i, val in enumerate(class_ratio):
        actuals = actuals + [i for x in range(int(val * multi))]
    preds = []
    for i in range(multi):
        preds += [class_ratio]
    if len(actuals) != len(preds):
        preds = preds[:len(actuals)]
    return (log_loss(actuals, preds))

# Determine statistics
logloss = calculate_log_loss(ratios,multi = 10000)

actuals = []
for i, val in enumerate(ratios):
    actuals = actuals + [i for x in range(int(val * 10000))]
from sklearn.metrics import confusion_matrix
import numpy as np
pred = [ratios,]*len(actuals)

SE = []
for i in range(len(pred[:])):
    row = [0, 0, 0, 0, 0]
    state_list = list(pred[i])
    max_value = max(state_list)
    SE.append(state_list.index(max_value))

CM = confusion_matrix(actuals, SE)
print(CM)
accuracy = np.diag(CM).sum() / CM.sum()
print('Accuracy: ', accuracy)

from sklearn.metrics import log_loss

print('Log Loss', log_loss(actuals, pred))

GT = []
for i in range(len(actuals)):
    row = [0, 0, 0, 0, 0]
    row[actuals[i]] = 1
    GT.append(row)
SE = []
for i in range(len(pred[:])):
    row = [0, 0, 0, 0, 0]
    state_list = list(pred[i])
    max_value = max(state_list)
    row[state_list.index(max_value)] = 1
    SE.append(row)

GT_sp = [GT[i][4] for i in range(len(GT))]
SE_sp = [pred[i][4] for i in range(len(pred))]
print('LL sp', log_loss(GT_sp, SE_sp))

from sklearn.metrics import roc_auc_score

print(roc_auc_score(GT, SE))