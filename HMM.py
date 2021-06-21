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
def train_models(link, lane, training_folder,save_folder, variables):
    folder = training_folder
    training_files = glob.glob(os.path.join(folder, '*.csv'))

    list_files = [x for x in training_files if str(int(link))+str(int(lane)) in x]
    #random.shuffle(list_files)
    counter = 0
    data = pd.DataFrame()
    for file in list_files:
        #if counter < 0.75*len(list_files):
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
#%%
    # scaler = MinMaxScaler()
    # scaler.fit(data[variables])
    #
    # from joblib import dump
    # scaler_file = save_folder + '\\norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin'
    # dump(scaler,scaler_file, compress = True)
    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

#%% Make different datasets
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

#%% Divide in different data sets

    data0 = X.loc[X.Previous_state == 0]
    data1 = X.loc[X.Previous_state == 1]
    data2 = X.loc[X.Previous_state == 2]
    data3 = X.loc[X.Previous_state == 3]
    data4 = X.loc[X.Previous_state == 4]

#%% Data analysis Data0
    def data_analysis(data0, data1, data2, data3, data4):
        a1 = data0[['TT', 'Current_state']]
        ax = a1.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data0[['Nb_stops', 'Current_state']]
        a2.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data0[['Shockwave', 'Current_state']]
        a3.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
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

        analysis_column(data0.TT)
        analysis_column(data0.Shockwave)

        analysis_column(data0.loc[data0.Current_state == 4].Shockwave)

        #%% Data analysis data1
        a1 = data1[['TT', 'Current_state']]
        ax = a1.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data1[['Nb_stops', 'Current_state']]
        a2.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data1[['Shockwave', 'Current_state']]
        a3.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
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

        analysis_column(data1.TT)
        analysis_column(data1.Shockwave)

        analysis_column(data1.loc[data1.Current_state == 4].Shockwave)

        #%% Data analysis data2
        a1 = data2[['TT', 'Current_state']]
        ax = a1.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data2[['Nb_stops', 'Current_state']]
        a2.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data2[['Shockwave', 'Current_state']]
        a3.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
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

        analysis_column(data2.TT)
        analysis_column(data2.Shockwave)

        analysis_column(data2.loc[data2.Current_state == 4].Shockwave)

        #%% Data analysis data3
        a1 = data3[['TT', 'Current_state']]
        ax = a1.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data3[['Nb_stops', 'Current_state']]
        a2.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data3[['Shockwave', 'Current_state']]
        a3.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
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

        analysis_column(data3.TT)
        analysis_column(data3.Shockwave)

        analysis_column(data3.loc[data3.Current_state == 4].Shockwave)

        #%% Data analysis data4
        a1 = data4[['TT', 'Current_state']]
        ax = a1.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a2 = data4[['Nb_stops', 'Current_state']]
        a2.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
                        showfliers=False, return_type = 'axes')
        a3 = data4[['Shockwave', 'Current_state']]
        a3.boxplot(by='Current_state', meanline = True, showmeans = True, showcaps = True, showbox = True,
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

        analysis_column(data4.TT)
        analysis_column(data4.Shockwave)

        analysis_column(data4.loc[data4.Current_state == 4].Shockwave)
        return
    #data_analysis(data0,data1,data2,data3,data4)
#%% Logistic Regression data0

    X_train = data0[variables]
    Y_train = data0[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr0= LogisticRegression(max_iter = 10000)
    LogRegr0.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state0_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr0, open(filename,'wb'))

    #%% Logistic Regression data1

    X_train = data1[variables]
    Y_train = data1[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr1= LogisticRegression(max_iter = 10000)
    LogRegr1.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state1_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr1, open(filename,'wb'))

    #%% Logistic Regression data2

    X_train = data2[variables]
    Y_train = data2[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr2= LogisticRegression(max_iter = 10000)
    LogRegr2.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state2_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr2, open(filename,'wb'))

    #%% Logistic Regression data 3

    X_train = data3[variables]
    Y_train = data3[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr3= LogisticRegression(max_iter = 10000)
    LogRegr3.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state3_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr3, open(filename,'wb'))

    #%% Logistic Regression data 4

    X_train = data4[variables]
    Y_train = data4[['Current_state']]
    from sklearn.linear_model import LogisticRegression
    LogRegr4= LogisticRegression(max_iter = 10000)
    LogRegr4.fit(X_train, Y_train.values.ravel())

    filename = save_folder + '\\LR_model_state4_' + str(int(link)) + str(int(lane)) + '.sav'
    pickle.dump(LogRegr4, open(filename,'wb'))

    return

#%% Load models

def test_models(link, lane, test_folder, save_folder, variables, xmin=None, xmax=None):
    import pickle

    filename= save_folder + '\\LR_model_state0_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr0 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state1_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr1 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state2_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr2 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state3_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr3 = pickle.load(open(filename, 'rb'))
    filename= save_folder + '\\LR_model_state4_' + str(int(link)) + str(int(lane)) + '.sav'
    LogRegr4 = pickle.load(open(filename, 'rb'))

    from joblib import load
    scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')
#%%
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


#%%
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

#print(get_transition_matrix(0.68232, 0.65625, 0.86690))

#%% Prepare test-data
    import glob
    import os
    import numpy as np
    import pandas as pd

    folder = test_folder
    training_files = glob.glob(os.path.join(folder, '*.csv'))
    list_files = [x for x in training_files if str(int(link)) + str(int(lane)) in x]
    # random.shuffle(list_files)
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
    cycles= sorted(set(data_test.Current_cycle.to_list()))
    y = {}
    for i in range(len(cycles)):
        state_now = data_test.loc[data_test.Current_cycle == cycles[i]].State.to_list()[0]
        y[cycles[i]] = state_now

    x = {}

    data_sample = data_test.sample(frac=0.1)

    for i in range(len(cycles)):
        data_current = data_sample.loc[data_sample.Current_cycle == cycles[i]]
        #data_current = data_current.sample(n=1)
        if not data_current.empty:
            TT = data_current.TT_on_link.mean()
            avg_stopping = data_current.Average_stopping_time.mean()
            nb_stops = data_current.Number_of_stops.mean()
            shockwave = data_current.Shockwave_intersection.mean()
            time_on_intersection = data_current.Time_on_intersection.mean()
            min_stop = data_current.Min_stop.mean()
            max_stop = data_current.Max_stop.mean()
            x[cycles[i]] = {'Current_cycle': cycles[i], 'TT_on_link': TT,
                            'Number_of_stops': nb_stops, 'Average_stopping_time':avg_stopping, 'Min_stop': min_stop,'Max_stop': max_stop,
                            'Time_on_intersection': time_on_intersection,'Shockwave_intersection': shockwave}
        else:
            if i == 0:
                x[cycles[0]] = {'Current_cycle': cycles[0], 'TT_on_link': 35,
                                'Number_of_stops': 0, 'Average_stopping_time': 0, 'Min_stop': 20, 'Max_stop': 0,
                                'Time_on_intersection':4.2,'Shockwave_intersection': 0.2}
            else:
                x[cycles[i]] = x[cycles[i-1]]
                x[cycles[i]]['Current_cycle'] = cycles[i] #Not correct yet

    for cycle in cycles:
        for key in x[cycle].copy().keys():
            if key not in variables + ['Current_cycle',]:
                del x[cycle][key]
    X_test = pd.DataFrame.from_dict({cycle: [x[cycle][key] for key in x[cycle].keys()]
                                     for cycle in x.keys()}, columns=['Current_cycle', ] + variables,
                                    orient='index')
    X_test[variables] = scaler.transform(X_test[variables])

#%%
    import time
    import numpy as np
    start_state = np.array([1,0,0,0,0]).T
    state_est = [start_state,]
    prev_state = start_state
    for i in range(len(cycles)):
        obs = X_test.loc[X_test.index == cycles[i]].to_numpy()
        #print(obs)
        #obs = ss.fit_transform(obs[['TT', 'Nb_stops','Shockwave']].to_numpy())
        TM = get_transition_matrix(obs[0][1:])
        new_state = np.dot(prev_state.T, TM)
        state_est =np.vstack([state_est, new_state])
        prev_state = new_state
    y_pred = []
    for state in state_est:
        estimated_state = np.argmax(state)
        y_pred.append(estimated_state)

    CM = pd.crosstab(np.array(y_pred[1:]), np.array(list(y.values())), colnames=['Actual'], rownames=['Predicted'])
    print(CM)
    accuracy = np.diag(CM).sum() / CM.to_numpy().sum()
    print('Accuracy: ', accuracy)

    from sklearn.metrics import log_loss
    try:
        print('Log Loss:', log_loss(list(y.values()), state_est[1:]))
    except:
        print('Not all states present in the model')


    #%% Plot outcome probabilities
    import matplotlib.pyplot as plt

    x0 = state_est[1:,0]
    x1 = state_est[1:,1]
    x2 = state_est[1:,2]
    x3 = state_est[1:,3]
    x4 = state_est[1:,4]
    y_axis = np.arange(len(x0))
    true_state = np.array(list(y.values()))
    #figs, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={'width_ratios':[30,1]})
    figs, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[30,1]})
    df = pd.DataFrame({'U':x0, 'O1': x1, 'O2': x2,'O3': x3,'Sp':x4})
    df.plot(kind = 'bar', ax = ax1, stacked = True, legend = False, color= ['forestgreen','yellowgreen','orange','orangered', 'darkred'])#,figsize=(12,14)) width = 0.8
    if xmin != None and xmax != None:
        ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([0,1])
    ax1.set_ylabel('Probability of being in state i [-]')
    ax1.axes.xaxis.set_visible(False)
    ax1.legend(bbox_to_anchor = (0.5,1.1), ncol = 5, loc = 'upper center', fancybox = True, shadow = True)
    y0,y1,y2,y3,y4 = [],[],[],[],[]

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
    df2 = pd.DataFrame({'U':y0, 'O1': y1, 'O2': y2,
                        'O3': y3,'Sp':y4})

    df2.plot(kind = 'bar', ax= ax2,stacked = True,legend=False, color= ['forestgreen','yellowgreen','orange','orangered', 'darkred']) #,figsize= (.5,14)
    if xmin != None and xmax != None:
        ax2.set_xlim([xmin,xmax])

    ax2.xaxis.set_ticks(np.arange(xmin, xmax + 1, 10))
    ax2.axes.yaxis.set_visible(False)
    ax2.set_xlabel('Time [minutes]')
    #ax1.set_title('State Estimation with Markov & Logistic Regression 10%')
    #plt.legend(loc= 'upper center', ncol = len(df.columns))
    #figs.subplots_adjust(bottom = 0.25)
    plt.show()

    #return veh_sample
