import numpy as np
import pandas as pd
import glob
import os

#%%
def downstream_state(training_folder, test_folder, model_folder, downstream_file_training, downstream_file_test, variables):
    score = {}
    final_score = {}
    for link in [44.0]:
        for lane in [2.0,3.0]:
            score[(link,lane)] = {}
            # Load regression models for this link-lane
            import pickle
            filename = model_folder + '\\LR_model_state_' + str(int(link)) + str(int(lane)) + '.sav'
            LogRegr = pickle.load(open(filename, 'rb'))


            # Load normalization
            from joblib import load
            scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

            # Transform data to test
            folder = training_folder
            training_files = glob.glob(os.path.join(folder, '*.csv'))
            list_files = [x for x in training_files if str(int(link)) + str(int(lane)) in x]
            # random.shuffle(list_files)
            counter = 0
            data = pd.DataFrame
            for file in list_files:
                if data.empty:
                    data = pd.read_csv(file, header=0)
                    counter += 1
                else:
                    data_new = pd.read_csv(file, header=0)
                    data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
                    data = pd.concat([data, data_new])
                    counter += 1
            lst_data = data['Min_stop'].to_list()
            for i in range(len(lst_data)):
                if lst_data[i] == 0:
                    lst_data[i] += 20
            data['Min_stop'] = lst_data
            cycles = sorted(set(data.Current_cycle.to_list()))
            y = {}
            for i in range(len(cycles)):
                state_now = data.loc[data.Current_cycle == cycles[i]].State.to_list()[0]
                y[cycles[i]] = state_now

            x = {}

            data_sample = data.sample(frac=0.1)

            for i in range(len(cycles)):
                data_current = data_sample.loc[data_sample.Current_cycle == cycles[i]]
                #data_current = data_current.sample(n=1)
                if not data_current.empty:
                    TT = data_current.TT_on_link.mean()
                    avg_stopping = data_current.Average_stopping_time.mean()
                    nb_stops = data_current.Number_of_stops.mean()
                    shockwave = data_current.Shockwave_intersection.mean()
                    max_stop = data_current.Max_stop.mean()
                    min_stop = data_current.Min_stop.mean()
                    time_on_intersection = data_current.Time_on_intersection.mean()
                    x[cycles[i]] = {'Current_cycle': cycles[i], 'TT_on_link': TT,
                                    'Number_of_stops': nb_stops, 'Average_stopping_time': avg_stopping, 'Min_stop':min_stop,
                                    'Max_stop': max_stop,
                                    'Time_on_intersection': time_on_intersection, 'Shockwave_intersection': shockwave}
                else:
                    if i == 0:
                        x[cycles[0]] = {'Current_cycle': cycles[0], 'TT_on_link': 35,
                                        'Number_of_stops': 0, 'Average_stopping_time': 0, 'Max_stop': 0, 'Min_stop': 20,
                                        'Time_on_intersection': 4.2, 'Shockwave_intersection': 0.2}
                    else:
                        x[cycles[i]] = x[cycles[i - 1]]
                        x[cycles[i]]['Current_cycle'] = cycles[i]  # Not correct yet

            for cycle in cycles:
                for key in x[cycle].copy().keys():
                    if key not in variables + ['Current_cycle', ]:
                        del x[cycle][key]
            X= pd.DataFrame.from_dict({cycle: [x[cycle][key] for key in x[cycle].keys()]
                                             for cycle in x.keys()}, columns=['Current_cycle', ] + variables,
                                            orient='index')
            X[variables] = scaler.transform(X[variables])
            import numpy as np

            X_var = X[variables]
            state_est = LogRegr.predict_proba(X_var)

            weights = np.array([0.05,0.05, 0.2, 0.4, 0.4])
            for T in range(len(cycles)):
                ests = state_est[T]
                score[(link,lane)][cycles[T]] = np.dot(weights, ests)


    weights_lane = np.array([0.60,0.4])
    for T in range(len(cycles)):
        scores = []
        for key in score.keys():
            if cycles[T] in score[key].keys():
                scores.append(score[key][cycles[T]])
            elif T!=0 and cycles[T-1] in score[key].keys():
                scores.append(score[key][cycles[T-1]])
            else:
                scores.append(0)
        final_score[cycles[T]] = np.dot(weights_lane, np.array(scores))


    with open(downstream_file_training, 'w') as f:
        for key in final_score.keys():
            f.write("%s,%s\n"%(key,final_score[key]))

#%%
    print('Test File')
    score = {}
    final_score = {}

    for link in [44.0]:
        for lane in [2.0,3.0]:
            score[(link,lane)] = {}
            # Load regression models for this link-lane
            import pickle
            filename = model_folder + '\\LR_model_state_' + str(int(link)) + str(int(lane)) + '.sav'
            LogRegr = pickle.load(open(filename, 'rb'))
            # Load normalization
            from joblib import load
            scaler = load('norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin')

            # Transform data to test
            folder = test_folder
            training_files = glob.glob(os.path.join(folder, '*.csv'))
            list_files = [x for x in training_files if str(int(link)) + str(int(lane)) in x]
            # random.shuffle(list_files)
            counter = 0
            data = pd.DataFrame()
            for file in list_files:
                if data.empty:
                    data = pd.read_csv(file, header=0)
                    counter += 1
                else:
                    data_new = pd.read_csv(file, header=0)
                    data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
                    data = pd.concat([data, data_new])
                    counter += 1
            lst_data = data['Min_stop'].to_list()
            for i in range(len(lst_data)):
                if lst_data[i] == 0:
                    lst_data[i] += 20
            data['Min_stop'] = lst_data
            cycles = sorted(set(data.Current_cycle.to_list()))
            y = {}
            for i in range(len(cycles)):
                state_now = data.loc[data.Current_cycle == cycles[i]].State.to_list()[0]
                y[cycles[i]] = state_now

            x = {}

            data_sample = data.sample(frac=0.1) # is not adapted for different penetration rates

            for i in range(len(cycles)):
                data_current = data_sample.loc[data_sample.Current_cycle == cycles[i]]
                #data_current = data_current.sample(n=1)
                if not data_current.empty:
                    TT = data_current.TT_on_link.mean()
                    avg_stopping = data_current.Average_stopping_time.mean()
                    nb_stops = data_current.Number_of_stops.mean()
                    shockwave = data_current.Shockwave_intersection.mean()
                    max_stop = data_current.Max_stop.mean()
                    min_stop = data_current.Min_stop.mean()
                    time_on_intersection = data_current.Time_on_intersection.mean()
                    x[cycles[i]] = {'Current_cycle': cycles[i], 'TT_on_link': TT,
                                    'Number_of_stops': nb_stops, 'Average_stopping_time': avg_stopping, 'Min_stop':min_stop,
                                    'Max_stop': max_stop,
                                    'Time_on_intersection': time_on_intersection, 'Shockwave_intersection': shockwave}
                else:
                    if i == 0:
                        x[cycles[0]] = {'Current_cycle': cycles[0], 'TT_on_link': 35,
                                        'Number_of_stops': 0, 'Average_stopping_time': 0, 'Max_stop': 0, 'Min_stop':20,
                                        'Time_on_intersection': 4.2, 'Shockwave_intersection': 0.2}
                    else:
                        x[cycles[i]] = x[cycles[i - 1]]
                        x[cycles[i]]['Current_cycle'] = cycles[i]  # Not correct yet

            for cycle in cycles:
                for key in x[cycle].copy().keys():
                    if key not in variables + ['Current_cycle', ]:
                        del x[cycle][key]
            X = pd.DataFrame.from_dict({cycle: [x[cycle][key] for key in x[cycle].keys()]
                                             for cycle in x.keys()}, columns=['Current_cycle', ] + variables,
                                            orient='index')
            X[variables] = scaler.transform(X[variables])
            import numpy as np

            X_var = X[variables]
            state_est = LogRegr.predict_proba(X_var)

            weights = np.array([0.05,0.05, 0.1, 0.4, 0.4])

            for T in range(len(cycles)):
                ests = state_est[T]
                score[(link,lane)][cycles[T]] = np.dot(weights, ests)

    weights_lane = np.array([0.60,0.4])
    for T in range(len(cycles)):
        scores =  []
        for key in score.keys():
            if cycles[T] in score[key].keys():
                scores.append(score[key][cycles[T]])
            elif T!=0 and cycles[T-1] in score[key].keys():
                scores.append(score[key][cycles[T-1]])
            else:
                scores.append(0)
        final_score[cycles[T]] = np.dot(weights_lane, np.array(scores))

    with open(downstream_file_test, 'w') as f:
        for key in final_score.keys():
            f.write("%s,%s\n"%(key,final_score[key]))