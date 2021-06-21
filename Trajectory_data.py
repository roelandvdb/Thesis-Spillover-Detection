from ReadTrajectories import *
from simulation_data import read_trafficlights, get_network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os
import glob

plt.rcParams.update({'figure.max_open_warning': 0})


def get_spillover2(trajectories, cycles, link, lane, signals):
    # Determines whether a spillover is present based on the traffic signals and the location at which the first
    # vehicle in the queue is stopped

    # Decide whether there is a spillover or not:
    spillovers = {l: 0 for l in cycles}
    sg = (int(signals['Config'][link][lane]['SC']), int(signals['Config'][link][lane]['Sgroup']))
    start_green = []
    start_amber = []
    for k in range(len(signals['State'][sg]['SIMSEC'])):
        if signals['State'][sg]['NewState'][k].strip() == 'green':
            start_green.append(signals['State'][sg]['SIMSEC'][k])
        if signals['State'][sg]['NewState'][k].strip() == 'amber':
            start_amber.append(signals['State'][sg]['SIMSEC'][k])
    so = []
    for vehID in trajectories.keys():
        xt_start, xt_points = get_number_of_stops(trajectories[vehID], [link,], 1)
        for s in range(len(xt_points)):
            stop = xt_points[s]
            if 0 < stop[1] <= 8:
                for k in range(len(start_green)-1):
                    if start_green[k] < start_amber[k]:
                        index_amber = k
                    else:
                        index_amber = k+1
                    if start_green[k]+6 <= stop[0] < start_green[k+1]-2:
                        so.append((xt_start[s][0], stop[0]))
                    elif xt_start[s][0] <= start_green[k] and stop[0] >= start_green[k+1]: # spillover lasts the entire time
                        print('Extra long spillover from', xt_start[s][0], 'until ', stop[0])
                        so.append((xt_start[s][0], stop[0]))
                    elif start_green[k] <= xt_start[s][0] < start_amber[index_amber]-1:
                        so.append((xt_start[s][0], stop[0]))
    # classify states as spillover or non-spillover
    for tup in so:
        for t in range(len(cycles)-1):
            if cycles[t] <= tup[1] < cycles[t+1]:
                if tup[0] < cycles[t+1] - 15:
                    spillovers[cycles[t]] = 'spillover' # if at the end of a cycle: don't include the cycle as spillover
                spillovers[cycles[t+1]] = 'spillover'
                if t != len(cycles) -2:
                    spillovers[cycles[t+2]] = 'spillover'
                print('spillover at: ', tup[0], 'until ', tup[1])

    for l in spillovers.keys():
        if spillovers[l] == 0:
            spillovers[l] = 'no spillover'
    return spillovers


def get_traffic_state(data, link, lane, cycles, spillovers):
    # classify the states based on the average number of stops. If spillover present --> overwrite to spillover
    state = {}
    for x in cycles:
        data_2 = data.loc[data.Current_cycle == x]

        if not data_2.empty:
            avg_stops = data_2.Number_of_stops.mean()
        else:
            avg_stops = 0
        if spillovers[x] == 'spillover':
             state[x] = 4 #4
        elif 0 <= avg_stops < 1:
            state[x] = 0
        elif 1 <= avg_stops < 2: #2
            state[x] = 1
        elif 2 <= avg_stops < 3: #3
            state[x] = 2
        else:
            state[x] = 3
    return state

def get_avg_time_of_passage(trajectory, links):
    # calculate the travel time of the vehicle over the links leading to the approach.
    # for longer approaches, it can consist of multiple links
    end_def = 0
    for k in range(len(trajectory['SIMSEC'])):
        if int(trajectory['LinkNo'][k]) in links:
            if k == 0 or int(trajectory['LinkNo'][k - 1]) not in links:
                start_time = trajectory['SIMSEC'][k]
            elif k == len(trajectory['SIMSEC']) - 1 or int(trajectory['LinkNo'][k + 1]) not in links:
                end_time = trajectory['SIMSEC'][k]
                end_def = 1
    if end_def == 0:
        end_time = trajectory['SIMSEC'][-1]
    return (start_time, end_time, end_time - start_time)

def get_number_of_stops(trajectory, links, addduration = 0):
    # get the number of stops of the vehicles during the approach, based on the driving speed of the vehicles.
    # If the duration of these stops needs to be known, give addduration value 1.
    network_file = 'Data/Simulation/network.xlsx'
    network = get_network(network_file)
    stops = []
    start_def = 0
    xt_points = []
    xt_start = []
    for k in range(len(trajectory['SIMSEC'])):
        if 0 <= trajectory['Speed'][k] < 2 and int(trajectory['LinkNo'][k]) in links:# stricter --> better shockwave estimation (<4)
            index = links.index(trajectory['LinkNo'][k])
            distance = 0
            if index == 0:
                distance = float(network[links[0]]['LENGTH2D']/1000)
            else:
                for i in range(index+1):
                    distance += float(network[links[i]]['LENGTH2D']/1000) # total driven length (necessary for longer approaches)

            # Determine stop duration
            if k == 0:
                start_time = trajectory['SIMSEC'][k]
                start_def = 1
            elif k == len(trajectory['SIMSEC'])-1:
                end_time = trajectory['SIMSEC'][k]
                if start_def == 0:
                    start_time = trajectory['SIMSEC'][k]
                if end_time - start_time > 2.0:
                    stops.append(end_time - start_time)
                    xt_points.append((end_time, distance - trajectory['POS'][k]))
                    xt_start.append((start_time, distance - trajectory['POS'][k]))
                start_def = 0
            elif trajectory['Speed'][k - 1] >= 2 or int(trajectory['LinkNo'][k + 1]) not in links:
                start_time = trajectory['SIMSEC'][k]
                start_def = 1
            elif trajectory['Speed'][k + 1] >= 2 or int(trajectory['LinkNo'][k + 1]) not in links:
                end_time = trajectory['SIMSEC'][k]
                if start_def == 0:
                    start_time = trajectory['SIMSEC'][k]
                if end_time - start_time > 2.0:
                    stops.append(end_time - start_time)
                    xt_points.append((end_time, distance - trajectory['POS'][k]))
                    xt_start.append((start_time, distance - trajectory['POS'][k]))
                start_def = 0

    if addduration == 0:
        if len(stops) == 0:
            return (0, 0, 0,0)
        return (len(stops), np.average(stops),max(stops),min(stops))
    else:
        if len(xt_points) == 0:
            return [(0,0)], [(0,0)] #(end_time, position)
        return xt_start, xt_points

def shockwave_speed(link=None, lane=None, trajectories=None, new_set = 0, sw_file = None):
    # Calculate the intersection of the interpolated shockwave with the stop line based on historical experienced shockwaves.
    # if new_set = 1: find in all trajectories a series of stopped vehicles and calculate their experienced shockwave
    if new_set == 1:
        stops = {}
        for veh in trajectories.keys():
            done = 0
            for k in range(len(trajectories[veh]['SIMSEC'])):
                if trajectories[veh]['LinkNo'][k] == link and trajectories[veh]['Lane'][k] == lane and done ==0:
                    xt_start, xt = get_number_of_stops(trajectories[veh], [link,], 1)
                    stops[veh] = {}
                    for val in xt:
                        stops[veh]['end_time'] = val[0]
                        stops[veh]['end_loc'] = val[1]
                    done = 1
        if len(stops)>0:
            df = pd.DataFrame.from_dict(stops, orient= 'index')
            df = df.loc[(df.end_time != 0) & (df.end_loc != 0)]

            #find cluster of data
            df = df.sort_values(['end_time', 'end_loc'], ascending=[True,True])
            lst_end_time = df.end_time.to_list()
            lst_end_loc = df.end_loc.to_list()
            lst = {0:[]}
            for k in range(1, len(lst_end_time)):
                if (lst_end_time[k-1] <lst_end_time[k] < lst_end_time[k-1]+3) and (lst_end_loc[k-1] > lst_end_loc[k] > lst_end_loc[k-1] - 10) :
                    lst[max(list(lst.keys()))].append((lst_end_time[k], lst_end_loc[k]))
                else:
                    lst[k] = [(lst_end_time[k],lst_end_loc[k]),]
            slopes = []
            for key in lst.keys():
                if len(lst[key]) > 3:
                    for i in range(len(lst[key])):
                        for j in range(i+1, len(lst[key])):
                            slopes.append(float((lst[key][j][1]-lst[key][i][1])/(lst[key][j][0] - lst[key][i][0]))) # in m/s
            if len(slopes) > 0:
                return statistics.mean(slopes)
            else:
                return 0
        else:
            return 0
    elif new_set == 0:
        slope = {}
        with open(sw_file, newline = '') as csvfile:
            spamreader = csv.reader(csvfile, delimiter= ',')
            for row in spamreader:
                if float(row[0]) not in slope.keys():
                    slope[float(row[0])] = {}
                slope[float(row[0])][float(row[1])] = float(row[2])
        return slope

# # Uncomment for saving new shockwave speeds; is based on a smaller sample of about 3 hours in the peak period,
# # but can still take several hours to run as every link is analysed separately, so replaced by safefile (new_set = 0).

# overview_map = "Data\Simulation\Shockwaves"
# traj_map = 'paths3'
# TL_file = 'Data\Simulation\Shockwaves\Deventer_TL.xlsx'
# network_file = 'Data\Simulation\\network.xlsx'
# network = get_network(network_file)
# shockwavespeeds = {}
# Overview_folder = os.path.join(overview_map, traj_map)
# csvfiles = glob.glob(os.path.join(Overview_folder, '*csv'))
# for link in network.keys():
#    shockwavespeeds[link] = {}
#    for lane in range(1,int(network[link]['NUMLANES'])+1):
#        trajectories = get_trajectories(link, lane, csvfiles)
#        shockwavespeeds[link][float(lane)] = shockwave_speed(link = link, lane =float(lane),trajectories = trajectories,new_set = 1)
# with open('Data\Simulation\Shockwaves\shockwavespeeds.csv', 'w', newline = '') as f:
#    w = csv.writer(f)
#    for key in shockwavespeeds.keys():
#        for lane in shockwavespeeds[key].keys():
#            w.writerow([key]+[lane]+[shockwavespeeds[key][lane]])

def find_intersection(xt_point, TL_location, sw_speed):
    # Linear extrapolation of the historical shockwaves
    x_int = TL_location
    t_int = xt_point[0]-(-xt_point[1])/sw_speed
    return (t_int,x_int)


def get_conn_links(link):
    # Get a list of links connecting to a certain link + the total travelled length
    TL_file = 'Data\Simulation\Shockwaves\Deventer_TL.xlsx'
    network_file = 'Data\Simulation\\network.xlsx'
    network = get_network(network_file)
    TL = read_trafficlights(TL_file)
    links = [link,]
    for l in network.keys():

        if network[l]['ISCONN'] == 1.0 and network[l]['TOLINK'] in links:
            lin = 0
            for key1 in TL['Config'].keys():
                for key2 in TL['Config'][key1].keys():
                    if l == TL['Config'][key1][key2]['Link']:
                        lin = 1
                    if network[l]['FROMLINK'] == TL['Config'][key1][key2]['Link']:
                        lin = 1
            if lin == 0:
                links.append(l)
                prev_link = network[l]['FROMLINK']
                lin = 0
                for key1 in TL['Config'].keys():
                    for key2 in TL['Config'][key1].keys():
                        if prev_link == TL['Config'][key1][key2]['Link']:
                            lin = 1
                if lin != 1:
                    links.append(prev_link)
    total_dist = 0
    for x in links:
        total_dist += network[x]['LENGTH2D']/1000
    return links, total_dist

def get_intersection_connector(trajectory, link):
    # get the connector that was used by a certain trajectory between link and the next link
    link_list = list(trajectory['LinkNo'])
    for L in range(len(link_list)-1):
        if link_list[L] == link:
            connector = link_list[L+1]
    return connector

#%%
def plot_link(trajectories, links, lane, startT, endT, TL_file, x_TL=None, prob = 0, state = None, shockwave = None, vehID = None):
    # plot the path of all trajectories on an approach (list of links) during a time period.
    # Possible to plot signal timings and shockwaves;
    # prob = 1 indictates that the total position needs to be included
    TL = read_trafficlights(TL_file)
    ax = plt.figure()
    if vehID == None:
        for vehID in trajectories.keys():
            link_k = []
            for k in range(0, len(trajectories[vehID]['SIMSEC'])):
                if trajectories[vehID]['LinkNo'][k] in links and startT <= trajectories[vehID]['SIMSEC'][k] <= endT\
                        and trajectories[vehID]['Lane'][k] == lane:
                    link_k.append(k)
            plotting_data_POS = []
            plotting_data_SIMSEC = []
            plotting_color = []
            for n in link_k:
                if prob == 1:
                    plotting_data_POS.append(trajectories[vehID]['TOTPOS'][n])
                else:
                    plotting_data_POS.append(trajectories[vehID]['POS'][n])
                plotting_data_SIMSEC.append(trajectories[vehID]['SIMSEC'][n])
                plotting_color.append(trajectories[vehID]['Speed'][n])

            if len(link_k) > 0:
                lanex = trajectories[vehID]['Lane'][link_k[-1]]
                if lanex == lane:
                    plt.scatter(plotting_data_SIMSEC, plotting_data_POS, c=plotting_color, cmap='RdYlGn', s=1)
            else:
                lanex = lane
    else:
        link_k = []
        for k in range(0, len(trajectories[vehID]['SIMSEC'])):
            if trajectories[vehID]['LinkNo'][k] in links and startT <= trajectories[vehID]['SIMSEC'][k] <= endT \
                    and trajectories[vehID]['Lane'][k] == lane:
                link_k.append(k)
        plotting_data_POS = []
        plotting_data_SIMSEC = []
        plotting_color = []
        for n in link_k:
            if prob == 1:
                plotting_data_POS.append(trajectories[vehID]['TOTPOS'][n])
            else:
                plotting_data_POS.append(trajectories[vehID]['POS'][n])
            plotting_data_SIMSEC.append(trajectories[vehID]['SIMSEC'][n])
            plotting_color.append(trajectories[vehID]['Speed'][n])

        if len(link_k) > 0:
            lanex = trajectories[vehID]['Lane'][link_k[-1]]
            if lanex == lane:
                plt.scatter(plotting_data_SIMSEC, plotting_data_POS, c=plotting_color, cmap='RdYlGn', s=1)
                # plt.title('lane: ' + str(lane))
        else:
            lanex = lane
    link_key = float(links[0])
    lane_key = lanex
    # plot the traffic lights
    if link_key in TL['Config'].keys() and lane_key in TL['Config'][link_key].keys():
        sg = (int(TL['Config'][link_key][lane_key]['SC']), int(TL['Config'][link_key][lane_key]['Sgroup']))
        if x_TL == None:
            x_TL = float(TL['Config'][link_key][lane_key]['At'])
        for k in range(len(TL['State'][sg]['SIMSEC']) - 1):
            if float(TL['State'][sg]['SIMSEC'][k]) >= startT and float(TL['State'][sg]['SIMSEC'][k]) <= endT:
                if TL['State'][sg]['NewState'][k] == ' red       ':
                    col = 'tab:red'
                elif TL['State'][sg]['NewState'][k] == ' amber     ':
                    col = 'tab:orange'
                else:
                    col = 'tab:green'
                xmin = int(float(TL['State'][sg]['SIMSEC'][k])) #! .strip()
                xmax = int(float(TL['State'][sg]['SIMSEC'][k]) + int(
                    float(TL['State'][sg]['DurationOfLastState'][k + 1])))
                if xmin >= startT and xmax <= endT:
                    simsec_TL = np.array(range(int(xmin), int(xmax)))
                    loc_TL2 = np.array([x_TL]*round(xmax-xmin))
                    plt.scatter(simsec_TL, loc_TL2, c = col, s = 3)
    if state != None:
        state = pd.DataFrame.from_dict(state)
        ax.pcolorfast(ax.get_xlim(),ax.get_ylim(),state.values[np.newaxis], cmap = 'RdYlGn', alpha =0.3)
    if shockwave != None:
        t0 = np.array([shockwave[0][0], shockwave[1][0]])
        x0 = np.array([shockwave[0][1], shockwave[1][1]])
        plt.plot(t0, x0, color='blue')
    plt.title('Trajectory on link '+str(int(link))+ ' and lane '+str(int(lane)))
    plt.xlabel('Time [s]')
    plt.ylabel('Travelled distance on the link [m]')
    plt.show()
    return


def plot_traj(trajectories, vehID, link, ymin=None, ymax=None):
    # plot the entire trajectory of a vehicle (over all links)
    for k in range(len(trajectories[vehID]['LinkNo']) - 1):
        if trajectories[vehID]['LinkNo'][k] != link and trajectories[vehID]['LinkNo'][k + 1] == link:
            plt.axhline(y=trajectories[vehID]['TOTPOS'][k])
        if trajectories[vehID]['LinkNo'][k] == link and trajectories[vehID]['LinkNo'][k + 1] != link:
            plt.axhline(y=trajectories[vehID]['TOTPOS'][k])
    Lcolors = np.array([int(trajectories[vehID]['Speed'][k] * 3.6) for k in range(len(trajectories[vehID]['Speed']))])
    plt.scatter(trajectories[vehID]['SIMSEC'], trajectories[vehID]['TOTPOS'], c=Lcolors, cmap='RdYlGn')
    if ymin != None:
        plt.ylim(ymin, ymax)
    plt.show()
    return
#%%
def training_trajectories(trajectories, link, lane, cycles,signals, shockwave):
    # Perform the taining of the trajectories on a link that end on a certain lane + assign to a cycle.
    # It includes the travel time, average stopping time, number of stops, min and max stop time, shockwave intersection,
    # and the time on the intersection

    column_names = ['VehID', 'Current_cycle', 'TT_on_link', 'Average_stopping_time', 'Number_of_stops','Max_stop','Min_stop',
                    'Shockwave_intersection', 'Time_on_intersection']
    training_vals = {}
    links, x_TL = get_conn_links(link)
    counter = 0
    network_file = 'Data\Simulation\\network.xlsx'
    network = get_network(network_file)
    length = network[link]['LENGTH2D']/1000
    for vehID in trajectories.keys():
        counter += 1
        if counter % 1000 == 0:
            print('Already ', counter, 'vehicles processed!')
        correct = 0
        kaas = []
        for k in range(len(trajectories[vehID]['SIMSEC'])):  #determine what sim secs are on link x
            if link == trajectories[vehID]['LinkNo'][k]:
                kaas.append(k)
        if len(kaas)>0 and trajectories[vehID]['Lane'][max(kaas)] == lane: # Determine if veh on correct lane
            correct = 1
        if correct == 1:
            # if wanted to plot the trajectories:
            #plot_traj(trajectories,vehID,link)
            # Calculate variables:
            time_of_passage = get_avg_time_of_passage(trajectories[vehID], links)
            average_duration_of_stops = get_number_of_stops(trajectories[vehID],links)  # (number of stops, average duration of stops)
            intersection_connector = get_intersection_connector(trajectories[vehID], link)
            time_on_intersection = get_avg_time_of_passage(trajectories[vehID],[intersection_connector,])
            # Determine shockwaves:
            link_key = float(links[0])
            lane_key = lane
            TL = signals
            if link_key in TL['Config'].keys() and lane_key in TL['Config'][link_key].keys():
                sg = (int(TL['Config'][link_key][lane_key]['SC']), int(TL['Config'][link_key][lane_key]['Sgroup']))
                TL_location = float(TL['Config'][link_key][lane_key]['At'])
            sw_speed = shockwave
            xt_start, xt_points = get_number_of_stops(trajectories[vehID], links, 1)
            shockwave_intersections = []
            for xt_point in xt_points:
                if xt_point != (0,0):
                    shockwave_intersections.append((find_intersection(xt_point, TL_location, sw_speed), xt_point[1]))
            differences = []

            for sw in shockwave_intersections:
                times = []
                for k in range(len(TL['State'][sg]['SIMSEC']) - 1):
                    if float(TL['State'][sg]['SIMSEC'][k]) < sw[0][0]+8 and TL['State'][sg]['NewState'][k].strip() == 'green':
                        times.append(float(TL['State'][sg]['SIMSEC'][k]))
                if len(times) > 0:

                    if sw[1] < length:
                        # Different weighting functions:
                        weight = -1.4 / (length ** 2) * sw[1] ** 2 + 1.1 / length * sw[1] + 0.8
                        # Other parabolic weight: weight = -3.2/(length**2) *sw[1]**2 + 3.2/length*sw[1] +0.2
                        # Linear weight:  weight = -0.5/length * sw[1] + 1
                    else:
                        weight = 0.5
                        # weight for other parabolic function: weight = 0.2
                    differences.append((sw[0][0] - max(times))*weight)

            if len(differences) > 0:

                diff = max(differences)
                if diff > 10:
                    pass
                    # if wanted: plot trajectories with higher difference than 10 seconds
                    #plot_traj(trajectories, vehID, link)
            else:
                diff = 0

            # Assign to cycle
            current_cycle = 0
            for T in range(len(cycles)-1):
                if cycles[T] <= time_of_passage[1] < cycles[T+1]:
                    current_cycle = cycles[T+1]

            # add variables
            training_vals[vehID] = [vehID, current_cycle, time_of_passage[2], average_duration_of_stops[1],
                                    average_duration_of_stops[0], average_duration_of_stops[2], average_duration_of_stops[3],
                                    diff, time_on_intersection[2]]
            #time.sleep(5)
    startT = 55000
    endT = 56000
    # If wanted: plot the trajectories over the link
    #plot_link(trajectories,links, lane, startT, endT, TL_file, x_TL, prob= 1)

    training_val = list(training_vals.values())

    X = []

    for k in range(len(training_val)):
        X.append(training_val[k])
    data= pd.DataFrame(X,columns = column_names)
    return data

#%% Example of the calculation (replaced by main-training for full dataset)
# link = 35.0
# lane = 3.0
# overview_map = 'Data/Simulation' #r"C:\Users\Roeland Vandenberghe\Documents\VLITS\Thesis\PythonFinalScripts\Data"
# traj_map = 'TrajectoriesPerVehicle/Day1'
# TL_file = 'Data/Simulation/Deventer_TL_HighRegime.xlsx' # careful: different TL-files are used for different regimes
# network_file = 'Data/Simulation/network.xlsx' #'Data/Simulation/network.xlsx' #'Data\Simulation\network.xslx'
# sw_file = 'Data/Simulation/Shockwaves/shockwavespeeds.csv'
# Overview_folder = os.path.join(overview_map, traj_map)
# csvfiles = glob.glob(os.path.join(Overview_folder, '*csv'))
# trajectories = get_trajectories(link, lane,csvfiles)
# cycles = get_cycles2(0,18000) # one period of 6 hours
# signals = read_trafficlights(TL_file)
# shockwave = shockwave_speed(new_set = 0, sw_file =sw_file)[link][lane] # calculate the shockwaves
# X = training_trajectories(trajectories, link, lane, cycles, signals, shockwave) # calculate attributes
# print('Searching for spillovers in cycle: ')
# spillover = get_spillover2(trajectories, cycles, link, lane, signals) # Determine presence of spillovers
# Y = get_traffic_state(X, link, lane, cycles, spillover)
#
# for cycle in Y.keys(): #if wanted: plot all spillover situations
#     if Y[cycle] == 4:
#         pass
#         plot_link(trajectories, [link,], lane,cycle-120, cycle+120, TL_file)
# lst = X.Current_cycle.to_list()
# state_list = []
# for elm in lst:
#     #print(elm, ' ', Y[elm])
#     state_list.append(Y[elm])
# X['State'] = state_list
# # if wanted: save in file, careful not to overwrite other files in data
# #X.to_csv('test_file.csv', index = False)

