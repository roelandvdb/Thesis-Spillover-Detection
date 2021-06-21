#from Trajectory_dataNew import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sn
import glob
import os
import random
pd.set_option('display.max_columns', 8)

def train_models(link, lane, training_folder,save_folder,variables):
    folder = training_folder
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
    lst_data = data['Min_stop'].to_list()
    for i in range(len(lst_data)):
        if lst_data[i] == 0:
            lst_data[i] += 20
    data['Min_stop'] = lst_data

#%% Scale the data

    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

#%% Exploratory data analysis
    def analyse_data(data):
        a1 = data[['TT_on_link', 'State']]
        ax = a1.boxplot(by='State', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data[['Average_stopping_time', 'State']]
        a2.boxplot(by='State', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data[['Number_of_stops', 'State']]
        a3.boxplot(by='State', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a4 = data[['Shockwave_intersection','State']]
        a4.boxplot(by='State', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        plt.show()

        def plotchart(col):
            fix, (ax1,ax2) = plt.subplots(1,2,figsize=(7,5))
            sn.boxplot(col, orient='v', ax = ax1)
            ax1.set_ylabel=col.name
            ax1.set_title('Box Plot of {}'.format(col.name))
            sn.distplot(col,ax=ax2)
            ax2.set_title('Distribution plot of {}'.format(col.name))
            plt.show()

        def analysis_column(col):
            print('Mean',format(col.mean()))
            print('Median',format(col.median()))
            plotchart(col)

        analysis_column(data.TT_on_link)

        analysis_column(data.Shockwave_intersection)
        analysis_column(data.loc[data.State==3].Shockwave_intersection)
        analysis_column(data.loc[data.State == 4].Shockwave_intersection)


        data.State.value_counts(normalize=True)

        corr = data.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()
        return
    #analyse_data(data)


#%% Data reworking from 100% penetration rate

    cycles= sorted(set(data.Current_cycle.to_list())) # too big gaps
    y = {}
    for i in range(len(cycles)):
        state_now = data.loc[data.Current_cycle == cycles[i]].State.to_list()[0]
        y[cycles[i]] = state_now

    X = data.to_numpy()
    columns = data.columns.to_list()
    columns = columns[1:]
    columns.append('Previous_state')
    X2 = np.zeros((len(X), 10))
    for i in range(len(X)):
        x = X[i][1:]
        if int(x[0] - 60) not in y.keys():
            X2[i] = np.append(x, 0)
        else:
            X2[i] = np.append(x, y[X[i][1] - 60])

    X = pd.DataFrame(data=X2, columns=columns)

    X.rename(columns={'State': 'Current_state'}, inplace=True)
    X[variables] = scaler.transform(X[variables])


#%% Train logistic regression model

    from sklearn.linear_model import LogisticRegression
    X_train = X[variables]
    Y_train = X[['Current_state']]
    LogRegr = LogisticRegression(max_iter = 10000)
    LogRegr.fit(X_train, Y_train)

    import pickle
    filename = save_folder + '\\LR_model_state_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr, open(filename, 'wb'))
    return
#%% Test data
def test_models(link,lane, test_folder, model_folder, variables,xmin = 200, xmax = 300, plot = 0):

    # Load models
    import pickle
    filename = model_folder + '\\LR_model_state_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr = pickle.load(open(filename, 'rb'))
    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

    # Load Data
    folder = test_folder
    training_files = glob.glob(os.path.join(folder, '*.csv'))
    list_files = [x for x in training_files if str(int(link)) + str(int(lane)) in x]
    counter = 0
    data_test = pd.DataFrame()
    for file in list_files:
        if data_test.empty:
            data_test = pd.read_csv(file, header=0)
            counter += 1
        else:
            data_new = pd.read_csv(file, header=0)
            data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
            data_test = pd.concat([data_test, data_new])
            counter += 1
    lst_data = data_test['Min_stop'].to_list()
    for i in range(len(lst_data)):
        if lst_data[i] == 0:
            lst_data[i] += 20
    data_test['Min_stop'] = lst_data
    # Make data with lower p
    cycles= sorted(set(data_test.Current_cycle.to_list()))
    y = {}
    for i in range(len(cycles)):
        state_now = data_test.loc[data_test.Current_cycle == cycles[i]].State.to_list()[0]
        y[cycles[i]] = state_now
    Y_test = pd.DataFrame.from_dict(y, columns = ['State'], orient = 'index')
    x = {}

    data_sample = data_test

    for i in range(len(cycles)):
        data_current = data_sample.loc[data_sample.Current_cycle == cycles[i]]
        data_current = data_current.sample(n=1)
        if not data_current.empty:
            TT = data_current.TT_on_link.mean()

            nb_stops = data_current.Number_of_stops.mean()
            shockwave = data_current.Shockwave_intersection.mean()
            avg_stopping = data_current.Average_stopping_time.mean()
            max_stop = data_current.Max_stop.mean()
            min_stop = data_current.Min_stop.mean()
            time_on_intersection = data_current.Time_on_intersection.mean()
            x[cycles[i]] = {'Current_cycle': cycles[i], 'TT_on_link': TT,
                            'Number_of_stops': nb_stops, 'Average_stopping_time': avg_stopping, 'Min_stop': min_stop,
                            'Max_stop': max_stop,
                            'Time_on_intersection': time_on_intersection,'Shockwave_intersection': shockwave}
        else:
            if i == 0:
                x[cycles[0]] = {'Current_cycle': cycles[0], 'TT_on_link': 35,
                                'Number_of_stops': 0, 'Average_stopping_time':0, 'Min_stop': 20, 'Max_stop': 0,
                                'Time_on_intersection': 4.2, 'Shockwave_intersection': 0.2}
            else:
                x[cycles[i]] = x[cycles[i-1]]
                x[cycles[i]]['Current_cycle'] = cycles[i]

    for cycle in cycles:
        for key in x[cycle].copy().keys():
            if key not in variables+['Current_cycle',]:
                del x[cycle][key]
    X_test = pd.DataFrame.from_dict({cycle: [x[cycle][key] for key in x[cycle].keys()]
                                for cycle in x.keys()}, columns = ['Current_cycle',]+variables,
                               orient = 'index')
    X_test[variables] = \
        scaler.transform(X_test[variables])

#%% Validation

    X_test = X_test[variables]
    Y_test= Y_test[['State']]

    predictions = LogRegr.predict(X_test)
    score = LogRegr.score(X_test,Y_test)

    y_pred = LogRegr.predict_proba(X_test)
    state_est = y_pred

    CM = pd.crosstab(predictions, Y_test.T.to_numpy()[0], colnames= ['Actual'], rownames=['Predicted'])
    print(CM)
    accuracy = np.diag(CM).sum() / CM.to_numpy().sum()
    print('Accuracy: ', accuracy)
    from sklearn.metrics import log_loss
    try:
        LL = log_loss(Y_test, y_pred)
        print(LL)
    except:
        print('Not the same number of states')
    GT = []
    for i in range(len(y.values())):
        row = [0, 0, 0, 0, 0]
        row[list(y.values())[i]] = 1
        GT.append(row)
    SE = []
    for i in range(len(state_est[:])):
        row = [0,0,0,0, 0]
        state_list = list(state_est[i])
        max_value = max(state_list)
        row[state_list.index(max_value)] = 1
        SE.append(row)
    from sklearn.metrics import classification_report
    print(classification_report(GT, SE))
    correct_spillover = []
    wrong_spillover = []
    wrong_spillover3 = []
    for i in range(len(y.values())):
        if list(y.values())[i] == 4:
            correct_spillover.append(state_est[i][4])
        elif list(y.values())[i] == 3:
            wrong_spillover3.append(state_est[i][4])
        else:
            wrong_spillover.append(state_est[i][4])
    import statistics
    print('During Spillover:', statistics.mean(correct_spillover))
    print('During non-spillover:', statistics.mean(wrong_spillover))
    print('During non-spillover in peak:', statistics.mean(wrong_spillover3))

    # plot results
    if plot == 1:
        import matplotlib.pyplot as plt

        x0 = state_est[1:, 0]
        x1 = state_est[1:, 1]
        x2 = state_est[1:, 2]
        x3 = state_est[1:, 3]
        x4 = state_est[1:, 4]
        y_axis = np.arange(len(x0))
        true_state = np.array(list(y.values()))
        figs, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [30, 1]})
        df = pd.DataFrame({'U': x0, 'O1': x1, 'O2': x2, 'O3': x3, 'Sp': x4})
        df.plot(kind='bar', ax=ax1, stacked=True, legend=False,
                color=['forestgreen', 'yellowgreen', 'orange', 'orangered', 'darkred'])
        if xmin != None and xmax != None:
            ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Probability of being in state i [-]')
        ax1.axes.xaxis.set_visible(False)
        ax1.legend(bbox_to_anchor=(0.5, 1.1), ncol=5, loc='upper center', fancybox=True, shadow=True)
        y0, y1, y2, y3, y4 = [], [], [], [], []

        for x in range(len(true_state)):
            if true_state[x] == 0:
                y0.append(1)
                y1.append(0)
                y2.append(0)
                y3.append(0)
                y4.append(0)
            elif true_state[x] == 1:
                y0.append(0)
                y1.append(1)
                y2.append(0)
                y3.append(0)
                y4.append(0)
            elif true_state[x] == 2:
                y0.append(0)
                y1.append(0)
                y2.append(1)
                y3.append(0)
                y4.append(0)
            elif true_state[x] == 3:
                y0.append(0)
                y1.append(0)
                y2.append(0)
                y3.append(1)
                y4.append(0)
            else:
                y0.append(0)
                y1.append(0)
                y2.append(0)
                y3.append(0)
                y4.append(1)
        df2 = pd.DataFrame({'U': y0, 'O1': y1, 'O2': y2,
                            'O3': y3, 'Sp': y4})

        df2.plot(kind='bar', ax=ax2, stacked=True, legend=False,
                 color=['forestgreen', 'yellowgreen', 'orange', 'orangered', 'darkred'])  # ,figsize= (.5,14)
        if xmin != None and xmax != None:
            ax2.set_xlim([xmin, xmax])

        ax2.xaxis.set_ticks(np.arange(xmin, xmax + 1, 10))
        ax2.axes.yaxis.set_visible(False)
        ax2.set_xlabel('Time [minutes]')

        plt.show()

    return accuracy, LL