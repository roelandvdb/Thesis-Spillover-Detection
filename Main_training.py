#%% This file will be the main file to run when new data is read, stored and processed;
# It is advised to run the file section per section (#%%)
from simulation_data import trajectory_files, read_trafficlights, get_network
from ReadTrajectories import get_trajectories, get_cycles2
import os
import glob
import warnings
from sklearn.preprocessing import MinMaxScaler
from Trajectory_data import *

warnings.filterwarnings('ignore')

#%% Make correct file names and folders; already happened for 'raw trajectories'
location = 'Data\Simulation\RawTrajectories'
files = glob.glob(os.path.join(location, '*.fzp'))
counter = 1
for file in files:
    new_name = location + '\\trajectories_day' + str(counter) + '.csv'
    counter += 1
    os.rename(file, new_name)

for c in range(1,21):
    paths = location + '\\Day' + str(c)
    if not os.path.exists(paths):
        os.mkdir(paths)

#%% Save trajectory files separately for each vehicle
def find_nb(file):
    for i in range(-8,-3):
        if file[i] == 'y':
            nb = file[i+1:-4]
            return nb

location = 'Data\Simulation\RawTrajectories'
files = glob.glob(os.path.join(location, 'trajectories_day'+'*.csv'))
for file in files:
    nb = find_nb(file)
    save_loc = "Data\Simulation\TrajectoriesPerVehicle\Day" + nb + "\\vehID"
    trajectory_files(file, save_loc,1)


#%% Assess the attributes of the model:
    # the following code is run for each regime separately, but only high regime included in this
    # example. The results for all regimes are available in the datamap.

TL_file = 'Data\Simulation\Deventer_TL_HighRegime.xlsx' # not always this TL-file for other regimes
network_file = 'Data\Simulation\\network.xlsx'
save_map = 'Data\Simulation\TrajectoriesPerVehicle'

# Training Data
for traj_map in ['Day1','Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8', 'Day9', 'Day10','Day11','Day12','Day13','Day14','Day15']:
    print('---', traj_map, '---')
    lanes = {35.0: [3.0,], 44.0:[1.0,2.0,3.0]}
    for link in [35.0, 44.0]:
        # read trajectory per vehicle and analyse the data
        network = get_network(network_file)
        signals = read_trafficlights(TL_file)
        overviewfolder = os.path.join(save_map, traj_map)
        csvfiles= glob.glob(os.path.join(overviewfolder, '*csv'))
        cycles = get_cycles2(0,18000)
        for lane in lanes[link]:
            print('Link:', link, 'Lane: ', lane)
            trajectories = get_trajectories(link, lane, csvfiles)
            #4 get training data for every trajectory
            sw_file = 'Data\Simulation\Shockwaves\shockwavespeeds.csv'
            shockwave = shockwave_speed(new_set=0, sw_file=sw_file)[link][lane]
            X = training_trajectories(trajectories, link, lane, cycles, signals, shockwave)
            print('Searching for spillovers in cycle: ')
            spillover = get_spillover2(trajectories, cycles, link, lane, signals)
            Y = get_traffic_state(X, link, lane, cycles, spillover)

            for cycle in Y.keys():
                if Y[cycle] == 4:
                    pass
                    # plot_link(trajectories, [link,], lane,cycle-120, cycle+120)
            lst = X.Current_cycle.to_list()
            state_list = []
            for elm in lst:
                # print(elm, ' ', Y[elm])
                state_list.append(Y[elm])
            X['State'] = state_list
            save_result = 'TrainingData\\training_data' + str(int(link)) + str(int(lane)) + '_' + traj_map + '.csv'
            X.to_csv(save_result, index=False)

# Test Data
for traj_map in ['Day16','Day17','Day18','Day19','Day20']:
    print('---', traj_map, '---')
    lanes = {35.0: [3.0,], 44.0:[1.0,2.0,3.0]}
    for link in [35.0, 44.0]:
        network = get_network(network_file)
        signals = read_trafficlights(TL_file)
        overviewfolder = os.path.join(save_map, traj_map)
        csvfiles= glob.glob(os.path.join(overviewfolder, '*csv'))
        cycles = get_cycles2(0,18000)
        for lane in lanes[link]:
            print('Link:', link, 'Lane: ', lane)
            trajectories = get_trajectories(link, lane, csvfiles)
            #4 get training data for every trajectory
            sw_file = 'Data\Simulation\Shockwaves\shockwavespeeds.csv'
            shockwave = shockwave_speed(new_set=0, sw_file=sw_file)[link][lane]
            X = training_trajectories(trajectories, link, lane, cycles, signals, shockwave)
            print('Searching for spillovers in cycle: ')
            spillover = get_spillover2(trajectories, cycles, link, lane, signals)
            Y = get_traffic_state(X, link, lane, cycles, spillover)

            for cycle in Y.keys():
                if Y[cycle] == 4:
                    pass
                    # plot_link(trajectories, [link,], lane,cycle-120, cycle+120)
            lst = X.Current_cycle.to_list()
            state_list = []
            for elm in lst:
                # print(elm, ' ', Y[elm])
                state_list.append(Y[elm])
            X['State'] = state_list
            save_result = 'TestData\\test_data' + str(int(link)) + str(int(lane)) + '_' + traj_map + '.csv'
            X.to_csv(save_result, index=False)

#%% Data Analysis:  analyse trajectory covariates
from DataAnalysis import *
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
variables = ['TT_on_link','Min_stop', 'Max_stop','Time_on_intersection',
             'Shockwave_intersection'] #,'Number_of_stops', 'Average_stopping_time'
# change according to researched lane
link = 35.0
lane = 3.0
print('----' + str(link) + ' and ' +  str(lane) + '----')
data_analysis(link,lane,training_folder,test_folder,variables)


#%% Normalize data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
variables = ['TT_on_link', 'Min_stop', 'Max_stop', 'Time_on_intersection',
             'Shockwave_intersection'] #,'Number_of_stops' ,'Average_stopping_time'
lanes = {35.0: [3.0,], 44.0:[1.0,2.0,3.0]}
for link in [35.0, 44.0]:
    for lane in lanes[link]:
        folder = training_folder
        training_files = glob.glob(os.path.join(folder, '*.csv'))
        list_files = [x for x in training_files if str(int(link)) + str(int(lane)) in x]
        test_files = glob.glob(os.path.join(test_folder, '*.csv'))
        list_files2 = [x for x in test_files if str(int(link)) + str(int(lane)) in x]

        counter = 0
        data = pd.DataFrame()
        for file in list_files + list_files2:
            if data.empty:
                data = pd.read_csv(file, header=0)
                counter += 1
            else:
                data_new = pd.read_csv(file, header=0)
                data_new['Current_cycle'] = data_new['Current_cycle'].astype(int) + counter * 18000
                data = pd.concat([data, data_new])
                counter += 1

        # Minimum Stop Time has value 0 if no stops --> wrong in attributes: should be 20 (otherwise no stop has high importance)
        lst_data = data['Min_stop'].to_list()
        for i in range(len(lst_data)):
            if lst_data[i] == 0:
                lst_data[i] += 20
        data['Min_stop'] = lst_data

        #Save normalisation file
        scaler = MinMaxScaler()
        scaler.fit(data[variables])
        from joblib import dump
        scaler_file = 'norm_scaler_' + str(int(link)) + str(int(lane)) + '.bin'
        dump(scaler, scaler_file, compress=True)


#%% Logistic Regression Only with n% penetration rate without downstream state
from Logistic_regression import *
import statistics

training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\LogRegOnly_wodownstream'
variables = ['TT_on_link', 'Min_stop', 'Max_stop','Time_on_intersection', 'Shockwave_intersection']

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,], 44.0:[1.0,2.0,3.0]}
for link in [35.0, 44.0]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' +  str(lane) + '----')
        train_models(link,lane,training_folder, save_folder,variables)

        # Test performance of the model
        acc_list = []
        logloss_list = []
        auc_list = []
        results = {}
        if link == 35.0:
            for pr in [0.05, 0.10,0.20,0.50]:
                for k in range(0,30):
                    acc, ll, AUC = test_models(link,lane,test_folder, save_folder, variables, 3150,3250, p = pr, plot = 0)
                    acc_list.append(acc)
                    logloss_list.append(ll)
                    auc_list.append(AUC)
                results[pr] = {'Accuracy': {'Average': statistics.mean(acc_list), 'SD': statistics.stdev(acc_list)},
                      'LogLoss': {'Average': statistics.mean(logloss_list), 'SD': statistics.stdev(logloss_list)},
                              'AUC': {'Average': statistics.mean(auc_list), 'SD': statistics.stdev(auc_list)}}
            print(results)

#%% Logistic Regression with n% penetration rate and downstream state
from Logistic_regression_wDownstreamState import *
from DownstreamStateLR import *
import statistics
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\LogRegOnly_wodownstream'
variables = ['TT_on_link', 'Min_stop','Max_stop','Time_on_intersection', 'Shockwave_intersection']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# Save downstream states:
downstream_file_training = save_folder + '\\downstream_state_training.csv'
downstream_file_test = save_folder + '\\downstream_state_test.csv'
downstream_state(training_folder, test_folder,save_folder,downstream_file_training,downstream_file_test, variables)

save_folder = 'Data\Models\LogRegOnly_wdownstream'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,]}
for link in [35.0,]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, downstream_file_training,variables)
        results = {}
        if link == 35.0:
            acc_list2 = []
            LL_list2 = []
            AUC_list2 = []
            Acc_spillover_list2 = []
            False_detection_list2 = []
            LL_sp_list2 = []
            for pr in np.arange(0.01, 1, 0.01):
                acc_temp = []
                LL_temp = []
                AUC_temp = []
                Acc_spillover_temp = []
                False_detection_temp = []
                LL_sp_temp = []
                for k in range(10):
                    acc, LL, AUC, AS, FD, LL_sp = test_models(link, lane, test_folder, save_folder,
                                                              downstream_file_test,variables,1930, 1990,pr, plot = 0)
                    acc_temp.append(acc)
                    LL_temp.append(LL)
                    AUC_temp.append(AUC)
                    Acc_spillover_temp.append(AS)
                    False_detection_temp.append(FD)
                    LL_sp_temp.append(LL_sp)
                acc_list2.append(statistics.mean(acc_temp))
                LL_list2.append(statistics.mean(LL_temp))
                AUC_list2.append(statistics.mean(AUC_temp))
                Acc_spillover_list2.append(statistics.mean(Acc_spillover_temp))
                False_detection_list2.append(statistics.mean(False_detection_temp))
                LL_sp_list2.append(statistics.mean(LL_sp_temp))

#%% Logistic Regression Only with 1 per cycle without downstream state
from Logistic_regression1PC import *
import statistics
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\LogRegOnly1PC_wodownstream'
variables = ['TT_on_link', 'Min_stop', 'Max_stop','Time_on_intersection', 'Shockwave_intersection']

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,], 44.0:[1.0,2.0,3.0]}
for link in [35.0, 44.0]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' +  str(lane) + '----')
        train_models(link,lane,training_folder, save_folder,variables)

#%% Logistic Regression Only with 1 per cycle with downstream state
from Logistic_regression1PC_wDownstreamState import *
from DownstreamStateLR1PC import *
import statistics
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\LogRegOnly1PC_wodownstream'
variables = ['TT_on_link', 'Min_stop','Max_stop','Time_on_intersection', 'Shockwave_intersection']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# Save downstream states:
downstream_file_training = save_folder + '\\downstream_state_training.csv'
downstream_file_test = save_folder + '\\downstream_state_test.csv'
downstream_state(training_folder, test_folder,save_folder,downstream_file_training,downstream_file_test, variables)

save_folder = 'Data\Models\LogRegOnly1PC_wdownstream'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,]}
for link in [35.0,]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, downstream_file_training,variables)
        results = {}
        if link == 35.0:
            acc_list = []
            LL_list = []
            AUC_list = []
            acc_spillover_list = []
            false_detection_list = []
            LL_sp_list = []
            for k in range(10):
                acc, LL, AUC,acc_spillover, false_detection, LL_sp = test_models(link, lane, test_folder, save_folder, downstream_file_test,variables,500, 560)
                acc_list.append(acc)
                LL_list.append(LL)
                AUC_list.append(AUC)
                acc_spillover_list.append(acc_spillover)
                false_detection_list.append(false_detection)
                LL_sp_list.append(LL_sp)
            results= {'Accuracy': {'Average': statistics.mean(acc_list), 'SD': statistics.stdev(acc_list)},
                       'LogLoss': {'Average': statistics.mean(LL_list), 'SD': statistics.stdev(LL_list)},
                        'AUC': {'Average': statistics.mean(AUC_list), 'SD': statistics.stdev(AUC_list)},
                      'Accuracy Spillover': {'Average':statistics.mean(acc_spillover_list), 'SD': statistics.stdev(acc_spillover_list)},
                      'False Detection': {'Average': statistics.mean(false_detection_list), 'SD': statistics.stdev(false_detection_list)},
                      'LL_sp ':{'Average': statistics.mean(LL_sp_list), 'SD':statistics.stdev(LL_sp_list)}}
print(results)

#%% HMM 1 vehicle per cycle without downstream state
from HMM1PC import *
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\MLR1PC_wodownstream'
variables = ['TT_on_link', 'Min_stop', 'Max_stop',
             'Time_on_intersection', 'Shockwave_intersection']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,], 44.0:[2.0,3.0]} # no 44-1 because the state does not experience spillovers --> adaptations to model necessary
for link in [35.0, 44.0]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, variables)
        if link == 35.0:
            test_models(link, lane, test_folder, save_folder, variables, 150,300)


#%% Markov Logistic Regression 1 vehicle per cycle with downstream state
from HMM1PC_wDownstreamState import *
from DownstreamStateHMM1PC import *
import statistics
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\MLR1PC_wodownstream'
variables = ['TT_on_link', 'Min_stop','Max_stop',
             'Time_on_intersection', 'Shockwave_intersection']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save downstream states:
downstream_file_training = save_folder + '\\downstream_state_training.csv'
downstream_file_test = save_folder + '\\downstream_state_test.csv'
downstream_state(training_folder, test_folder,save_folder,downstream_file_training,downstream_file_test, variables)

# Train full model
save_folder = 'Data\Models\MLR1PC_wdownstream_AllRegimes2'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,]}
for link in [35.0,]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, downstream_file_training,variables)
        # # Uncomment to run for multiple penetration rates, plot = 1 for plotting results
        # acc_list = []
        # LL_list = []
        # AUC_list = []
        # acc_spillover_list = []
        # false_detection_list = []
        # LL_sp_list = []
        # if link == 35.0:
        #     for k in range(10):

        acc, LL,AUC, AS, FD, LL_sp = test_models(link, lane, test_folder, save_folder, downstream_file_test,variables,1930, 1990, plot = 0)
        #         acc_list.append(acc)
        #         LL_list.append(LL)
        #         AUC_list.append(AUC)
        #         acc_spillover_list.append(AS)
        #         false_detection_list.append(FD)
        #         LL_sp_list.append(LL_sp)
        # results = {'Accuracy': {'Average': statistics.mean(acc_list), 'SD': statistics.stdev(acc_list)},
        #            'LogLoss': {'Average': statistics.mean(LL_list), 'SD': statistics.stdev(LL_list)},
        #            'AUC': {'Average': statistics.mean(AUC_list), 'SD': statistics.stdev(AUC_list)},
        #            'Accuracy Spillover': {'Average': statistics.mean(acc_spillover_list),
        #                                   'SD': statistics.stdev(acc_spillover_list)},
        #            'False Detection': {'Average': statistics.mean(false_detection_list),
        #                                'SD': statistics.stdev(false_detection_list)},
        #            'LL_sp': {'Average':statistics.mean(LL_sp_list), 'SD':statistics.stdev(LL_sp_list)}}
#print(results)

#%% Markov Logistic Regression 10% penetration without downstream state
from HMM import *
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\MLR10p_wodownstream'
variables = ['TT_on_link', 'Min_stop', 'Max_stop',
             'Time_on_intersection', 'Shockwave_intersection']
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,], 44.0:[2.0,3.0]}
for link in [35.0, 44.0]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, variables)
        # if link == 35.0:
        #     test_models(link, lane, test_folder, save_folder, variables, 200,300)


#%% Markov Logistic Regression 10% penetration with downstream state
from HMM_wDownstreamState import *
from DownstreamStateHMM import *
import matplotlib.pyplot as plt
import statistics
training_folder = 'Data\Simulation\TrainingData'
test_folder = 'Data\Simulation\TestData'
save_folder = 'Data\Models\MLR10p_wodownstream'
variables = ['TT_on_link', 'Min_stop', 'Max_stop',
             'Time_on_intersection', 'Shockwave_intersection'] #'Max_stop',
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save downstream states:
downstream_file_training = save_folder + '\\downstream_state_training.csv'
downstream_file_test = save_folder + '\\downstream_state_test.csv'
downstream_state(training_folder, test_folder,save_folder,downstream_file_training,downstream_file_test, variables)

# Train Full model
save_folder = 'Data\Models\MLR10p_wdownstream'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
lanes = {35.0: [3.0,]}
for link in [35.0,]:
    for lane in lanes[link]:
        print('----' + str(link) + ' and ' + str(lane) + '----')
        train_models(link,lane,training_folder, save_folder, downstream_file_training,variables)
        if link == 35.0:
            ## uncomment this for an overview of the results per penetration rate; otherwise only one run
            # results = {}
            # acc_list = []
            # logloss_list = []
            # auc_list = []
            # acc_spillover_list = []
            # false_detection_list = []
            # LL_sp_list = []
            # #for pr in np.arange(0.01,1, 0.01):
            #     acc_temp = []
            #     logloss_temp = []
            #     auc_temp = []
            #     acc_spillover_temp = []
            #     false_detection_temp = []
            #     LL_sp_temp = []
            #     for k in range(0,10):
            pr = 0.05 # Delete this
            acc, ll, AUC, AS, FD, LL_sp = test_models(link, lane, test_folder, save_folder, downstream_file_test,variables,3170,3230, p= pr, plot = 0) # 2 tabs indent
            #         acc_temp.append(acc)
            #         logloss_temp.append(ll)
            #         auc_temp.append(AUC)
            #         acc_spillover_temp.append(AS)
            #         false_detection_temp.append(FD)
            #         LL_sp_temp.append(LL_sp)
            #     acc_list.append(statistics.mean(acc_temp))
            #     logloss_list.append(statistics.mean(logloss_temp))
            #     auc_list.append(statistics.mean(auc_temp))
            #     acc_spillover_list.append(statistics.mean(acc_spillover_temp))
            #     false_detection_list.append(statistics.mean(false_detection_temp))
            #     LL_sp_list.append(statistics.mean(LL_sp_temp))

print('Accuracy:', acc_list)
print('Log loss:', logloss_list)
print('AUC:', auc_list)

test_models(link, lane, test_folder, save_folder, downstream_file_test, variables, xmin= 3170, xmax = 3230, p = 0.1, plot = 1)
# # UNCOMMENT: plots of performance for different penetration rates of the LR and HMM models
# plt.plot(np.arange(1, 100, 1), acc_list, label='M2')
# plt.plot(np.arange(1, 100, 1), acc_list2, label='M1')
# plt.grid()
# plt.legend()
# plt.ylabel('Accuracy [-]')
# plt.xlabel('Penetration rate [%]')
# plt.xlim(0,100)
# plt.ylim(0,1)
# plt.savefig(r'C:\Users\Roeland Vandenberghe\Documents\VLITS\Thesis\Report\Figures\Accuracy_comparison.png')
# plt.show()
#
# plt.plot(np.arange(1, 100, 1), acc_list, label='M2')
# plt.plot(np.arange(1, 100, 1), acc_list2, label='M1')
# plt.grid()
# plt.legend()
# plt.ylabel('Log Loss [-]')
# plt.xlabel('Penetration rate [%]')
# plt.xlim(0,100)
# plt.ylim(0,1.7)
# plt.savefig(r'C:\Users\Roeland Vandenberghe\Documents\VLITS\Thesis\Report\Figures\Logloss_comparison.png')
# plt.show()


#%% Read the models of the HMM and look at attributes
import pickle
save_folder = 'Data\Models\MLR10p_wdownstream'
link = 35.0
lane = 3.0
filename= save_folder + '\\LR_model_state0_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
LogRegr02 = pickle.load(open(filename, 'rb'))
filename= save_folder + '\\LR_model_state1_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
LogRegr12 = pickle.load(open(filename, 'rb'))
filename= save_folder + '\\LR_model_state2_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
LogRegr22 = pickle.load(open(filename, 'rb'))
filename= save_folder + '\\LR_model_state3_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
LogRegr32 = pickle.load(open(filename, 'rb'))
filename= save_folder + '\\LR_model_state4_downstream_' + str(int(link)) + str(int(lane)) + '.sav'
LogRegr42 = pickle.load(open(filename, 'rb'))
