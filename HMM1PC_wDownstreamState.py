#%% Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sn
import pickle
import glob
import os
import random
pd.set_option('display.max_columns', 8)
#%%
def train_models(link,lane, training_folder, save_folder, downstream_file, variables):
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
    # Minimum Stop Time has value 0 if no stops --> wrong: should be 20
    lst_data = data['Min_stop'].to_list()
    for i in range(len(lst_data)):
        if lst_data[i] == 0:
            lst_data[i] += 20
    data['Min_stop'] = lst_data

    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

    #%% Make different datasets
    cycles= sorted(set(data.Current_cycle.to_list()))
    y = {}
    for i in range(len(cycles)):
        state_now = data.loc[data.Current_cycle == cycles[i]].State.to_list()[0]
        y[cycles[i]] = state_now

    import csv
    ds = {}
    with open(downstream_file, mode='r') as f:
        reader = csv.reader(f)
        for line in reader:
            ds[int(line[0])] = float(line[1]) #Downstream state

    X = data.to_numpy()
    columns = data.columns.to_list()
    columns = columns[1:]
    columns.append('Previous_state')
    columns.append('Downstream_state')
    X2 = np.zeros((len(X), 11))
    for i in range(len(X)):
        x = X[i][1:]
        if int(x[0] - 60) not in y.keys():
            x = np.append(x, 0)
        else:
            x = np.append(x, y[X[i][1] - 60])
        if x[0] in ds.keys():
            X2[i] = np.append(x, ds[x[0]])
        else:
            X2[i] = np.append(x, 0)

    X = pd.DataFrame(data=X2, columns=columns)
    X.rename(columns={'State': 'Current_state'}, inplace=True)
    X[variables] = scaler.transform(X[variables])

    #%% Divide in different data sets

    data0 = X.loc[X.Previous_state == 0]
    data1 = X.loc[X.Previous_state == 1]
    data2 = X.loc[X.Previous_state == 2]
    data3 = X.loc[X.Previous_state == 3]
    data4 = X.loc[X.Previous_state == 4]

    # Data analysis per dataset
    def data_analysis(data0,data1, data2, data3, data4):
        # data0
        for variable in variables:
            a = data0[[variable, 'Current_state']]
            ax = a.boxplot(by='Current_state', meanline=True, showmeans=True, showcaps=True, showbox=True,
                            showfliers=True, return_type='axes')
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

        corr = data0.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()


        #%% Data analysis data1
        for variable in variables:
            a = data1[[variable, 'Current_state']]
            ax = a.boxplot(by='Current_state', meanline=True, showmeans=True, showcaps=True, showbox=True,
                            showfliers=True, return_type='axes')
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

        corr = data1.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()


        #%% Data analysis data2
        vars = variables + ['Downstream_state',]
        for variable in vars:
            a = data2[[variable, 'Current_state']]
            ax = a.boxplot(by='Current_state', meanline=True, showmeans=True, showcaps=True, showbox=True,
                            showfliers=True, return_type='axes')
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

        corr = data2.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()

        #%% Data analysis data3
        vars = variables + ['Downstream_state',]
        for variable in vars:
            a = data3[[variable, 'Current_state']]
            ax = a.boxplot(by='Current_state', meanline=True, showmeans=True, showcaps=True, showbox=True,
                            showfliers=True, return_type='axes')
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

        corr = data3.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()

        #%% Data analysis data4
        vars = variables + ['Downstream_state',]
        for variable in vars:
            a = data4[[variable, 'Current_state']]
            ax = a.boxplot(by='Current_state', meanline=True, showmeans=True, showcaps=True, showbox=True,
                            showfliers=True, return_type='axes')
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

        corr = data4.corr(method='spearman')
        plt.figure(figsize=(15,15))
        sn.heatmap(corr, vmax = .8, linewidths=0.01, square = True, annot=True, cmap= 'RdBu', linecolor='black')
        plt.show()
        return

    #%% Logistic Regression data0
    vars = variables + ['Downstream_state']
    X_train = data0[vars]
    Y_train = data0[['Current_state']]

    from sklearn.linear_model import LogisticRegression
    LogRegr0= LogisticRegression(max_iter = 10000)
    LogRegr0.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state0_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr0, open(filename,'wb'))

    #%% Logistic Regression data1
    vars = variables + ['Downstream_state']
    X_train = data1[vars]
    Y_train = data1[['Current_state']]

    from sklearn.linear_model import LogisticRegression
    LogRegr1= LogisticRegression(max_iter = 10000)
    LogRegr1.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state1_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr1, open(filename,'wb'))

    #%% Logistic Regression data2
    vars = variables + ['Downstream_state']
    X_train = data2[vars]
    Y_train = data2[['Current_state']]

    from sklearn.linear_model import LogisticRegression
    LogRegr2= LogisticRegression(max_iter = 10000)
    LogRegr2.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state2_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr2, open(filename,'wb'))

    #%% Logistic Regression data 3

    X_train = data3[vars]
    Y_train = data3[['Current_state']]

    from sklearn.linear_model import LogisticRegression
    LogRegr3= LogisticRegression(max_iter=  10000)
    LogRegr3.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state3_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr3, open(filename,'wb'))

    #%% Logistic Regression data 4
    vars = variables + ['Downstream_state']
    X_train = data4[vars]
    Y_train = data4[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr4= LogisticRegression(max_iter = 10000)
    LogRegr4.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state4_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr4, open(filename,'wb'))
    return


def test_models(link,lane, test_folder, save_folder, downstream_folder, variables,xmin= None, xmax=None, plot = 0):
    # Load models
    import pickle
    filename= save_folder + '\\LR_model_state0_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr0 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state1_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr1 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state2_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr2 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state3_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr3 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state4_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr4 = pickle.load(open(filename, 'rb'))

    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

    # Make Transition matrices
    states = [0,1,2,3,4]
    not_in_regr = {state: [] for state in states}

    for state in states:
        if state not in LogRegr0.classes_:
            not_in_regr[0].append(state)
        if state not in LogRegr1.classes_:
            not_in_regr[1].append(state)
        if state not in LogRegr2.classes_:
            not_in_regr[2].append(state)
        if state not in LogRegr3.classes_:
            not_in_regr[3].append(state)
        if state not in LogRegr4.classes_:
            not_in_regr[4].append(state)
    def get_transition_matrix(obs):
        transition_matrix = np.zeros((len(states), len(states)))
        for from_state in states:
            if from_state == 0:
                regr = LogRegr0
            elif from_state == 1:
                regr = LogRegr1
            elif from_state == 2:
                regr = LogRegr2
            elif from_state == 3:
                regr = LogRegr3
            else:
                regr = LogRegr4
            prediction = regr.predict_proba(np.array([obs]))[0]
            index = 0
            for to_state in states:
                if to_state not in not_in_regr[from_state]:
                    transition_matrix[from_state][to_state] = prediction[index]
                    index +=1
                else:
                    transition_matrix[from_state][to_state] = 0
        return transition_matrix

    #%% Prepare test-data

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
    # Minimum Stop Time has value 0 if no stops --> wrong: should be 20
    lst_data = data_test['Min_stop'].to_list()
    for i in range(len(lst_data)):
        if lst_data[i] == 0:
            lst_data[i] += 20
    data_test['Min_stop'] = lst_data

    cycles= sorted(set(data_test.Current_cycle.to_list()))
    y = {}
    for i in range(len(cycles)):
        state_now = data_test.loc[data_test.Current_cycle == cycles[i]].State.to_list()[0]
        y[cycles[i]] = state_now

    x = {}

    data_sample = data_test

    for i in range(len(cycles)):
        data_current = data_sample.loc[data_sample.Current_cycle == cycles[i]]
        data_current = data_current.sample(n=1) #1 vehicle per cycle
        if not data_current.empty:
            TT = data_current.TT_on_link.mean()
            avg_stopping = data_current.Average_stopping_time.mean()
            nb_stops = data_current.Number_of_stops.mean()
            shockwave = data_current.Shockwave_intersection.mean()
            max_stop = data_current.Max_stop.mean()
            min_stop = data_current.Min_stop.mean()
            time_on_intersection = data_current.Time_on_intersection.mean()
            x[cycles[i]] = {'Current_cycle': cycles[i], 'TT_on_link': TT,
                            'Number_of_stops': nb_stops, 'Average_stopping_time':avg_stopping,'Min_stop': min_stop,
                            'Max_stop': max_stop,'Time_on_intersection': time_on_intersection,
                            'Shockwave_intersection': shockwave}
        else:
            if i == 0:
                x[cycles[0]] = {'Current_cycle': cycles[0], 'TT_on_link': 35,
                                'Number_of_stops': 0, 'Average_stopping_time': 0, 'Min_stop': 20,'Max_stop': 0,
                                'Time_on_intersection':4.2,'Shockwave_intersection': 0.2}
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
    import csv
    ds = {}
    downstream_file = downstream_folder
    with open(downstream_file,mode= 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            ds[int(line[0])] = float(line[1])
    dstate = []
    for cycle in cycles:
        if cycle in ds.keys():
            dstate.append(ds[cycle])
        if cycle not in ds.keys():
            dstate.append(0)

    X_test['Downstream_state'] = dstate

    X_test[variables] = scaler.transform(X_test[variables])

    print('Penetration rate:', len(cycles)/len(data_test))
    #%% Run model
    import numpy as np
    start_state = np.array([1,0,0,0,0]).T
    state_est = [start_state,]
    prev_state = start_state
    y_pred = []
    for i in range(len(cycles)):
        obs = X_test.loc[X_test.index == cycles[i]].to_numpy()
        TM = get_transition_matrix(obs[0][1:]) # Make transition matrix
        new_state = np.dot(prev_state.T, TM)
        state_est =np.vstack([state_est, new_state])
        prev_state = new_state

    # Analyse results
        # Accuracy
    y_pred = []
    for state in state_est:
        estimated_state = np.argmax(state)
        y_pred.append(estimated_state)

    CM = pd.crosstab(np.array(y_pred[1:]), np.array(list(y.values())), colnames= ['Actual'], rownames=['Predicted'])
    print(CM)
    accuracy = np.diag(CM).sum() / CM.to_numpy().sum()
    print('Accuracy: ', accuracy)
    CM = CM.to_numpy()
    accuracy_Spillover = CM[4][4]/(CM[4][4]+CM[3][4])
    false_detection = (CM[4][3]+CM[4][2]+CM[4][1]+CM[4][0])/(CM[4][4]+CM[4][3]+CM[4][2]+CM[4][1]+CM[4][0])
    print('AS:',accuracy_Spillover)
    print('FD:',false_detection)
        # Log Loss
    from sklearn.metrics import log_loss
    try:
        LL = log_loss(list(y.values()), state_est[1:])
        print(LL)
    except:
        print('Not all states present')

        # make list of each state separately
    GT = []
    for i in range(len(y.values())):
        row = [0, 0, 0, 0, 0]
        row[list(y.values())[i]] = 1
        GT.append(row)
    SE = []
    for i in range(len(state_est[1:])):
        row = [0,0,0,0, 0]
        state_list = list(state_est[i])
        max_value = max(state_list)
        row[state_list.index(max_value)] = 1
        SE.append(row)

        # Classification Report
    from sklearn.metrics import classification_report
    print(classification_report(GT, SE))

        # False detections
    correct_spillover = []
    wrong_spillover = []
    wrong_spillover3 = []
    for i in range(len(y.values())):
        if list(y.values())[i] == 4:
            correct_spillover.append(state_est[i+1][4])
        elif list(y.values())[i] == 3:
            wrong_spillover3.append(state_est[i+1][4])
        else:
            wrong_spillover.append(state_est[i+1][4])

        # ROC-AUC score
    import statistics
    from sklearn.metrics import roc_auc_score
    AUC = roc_auc_score(GT, state_est[1:])
    print(AUC)
    import matplotlib.pyplot as plt
    df = pd.DataFrame()

    df['GroundTruth'] = list(y.values())
    df['U'] = state_est[1:,0]
    df['O1'] = state_est[1:,1]
    df['O2'] = state_est[1:,2]
    df['O3'] = state_est[1:,3]
    df['Sp'] = state_est[1:,4]

        # Plot boxplot of the states predictions
    # a = df[['U', 'GroundTruth']]
    # ax = a.boxplot(by='GroundTruth', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                 showfliers=True, return_type='axes')
    # plt.show()
    # a = df[['O1', 'GroundTruth']]
    # ax = a.boxplot(by='GroundTruth', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                 showfliers=True, return_type='axes')
    # plt.show()
    # a = df[['O2', 'GroundTruth']]
    # ax = a.boxplot(by='GroundTruth', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                 showfliers=True, return_type='axes')
    # plt.show()
    # a = df[['O3', 'GroundTruth']]
    # ax = a.boxplot(by='GroundTruth', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                 showfliers=True, return_type='axes')
    # plt.show()
    # a = df[['Sp', 'GroundTruth']]
    # ax = a.boxplot(by='GroundTruth', meanline=True, showmeans=True, showcaps=True, showbox=True,
    #                 showfliers=True, return_type='axes')
    # plt.show()

        # Log Loss for spillover only
    df = pd.DataFrame()
    df['GroundTruth'] = list(y.values())
    df['U'] = state_est[1:,0]
    df['O1'] = state_est[1:,1]
    df['O2'] = state_est[1:,2]
    df['O3'] = state_est[1:,3]
    df['Sp'] = state_est[1:,4]
    GT_sp = [GT[i][4] for i in range(len(GT))]
    SE_sp = df['Sp'].to_list()
    LL_sp = log_loss(GT_sp, SE_sp)
    print('LL_SP', log_loss(GT_sp, SE_sp))

        # Predicted probabilities during spillover - nonspillover
    print('During Spillover: mean: ', statistics.mean(correct_spillover), ', standard deviation: ', statistics.stdev(correct_spillover))
    print('During non-spillover:', statistics.mean(wrong_spillover), ', standard deviatoin: ', statistics.stdev(wrong_spillover))
    print('During non-spillover in peak:', statistics.mean(wrong_spillover3), ', standard deviation: ', statistics.stdev(wrong_spillover3))


    #%% Plot outcome probabilities
    if plot == 1:
        import matplotlib.pyplot as plt

        x0 = state_est[1:, 0]
        x1 = state_est[1:, 1]
        x2 = state_est[1:, 2]
        x3 = state_est[1:, 3]
        x4 = state_est[1:, 4]
        y_axis = np.arange(len(x0))
        true_state = np.array(list(y.values()))
        figs, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [25, 1]})
        df = pd.DataFrame({'U': x0, 'O1': x1, 'O2': x2, 'O3': x3, 'Sp': x4})
        df.plot(kind='bar', ax=ax1, stacked=True, legend=False,
                color=['forestgreen', 'yellowgreen', 'orange', 'orangered', 'darkred'])  # ,figsize=(12,14)) width = 0.8
        if xmin != None and xmax != None:
            ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Probability of being in state i [-]')
        ax1.axes.xaxis.set_visible(False)
        ax1.legend(bbox_to_anchor=(0.5, 1.11), ncol=5, loc='upper center', fancybox=True, shadow=True)
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
    return accuracy, LL, AUC, accuracy_Spillover, false_detection, LL_sp