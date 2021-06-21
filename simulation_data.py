#############################################################################
######################## ANALYSIS OF SIMULATION DATA ########################
# Author: Roeland Vandenberhge
# Goal: in this file, data from the VISSIM-simulation is transformed in preparation of
#       SPM-analysis;

# Packages
import xlrd
import pandas as pd
import csv
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import random
import time

# Where to get data:

#HDD
#location = r"D:\Thesis\SimulationOutput2503\trajectories.csv"
#save_loc = r"D:\Thesis\SimulationOutput2503\paths\vehID"
#TL_file = r"D:\Thesis\SimulationOutput2503\Deventer_TL.xlsx"
#network_file = r"D:\Thesis\SimulationOutput2503\network.xlsx"
#save_map = "D:\Thesis\SimulationOutput2503\paths"

def save_files(data_per_veh, save_loc):
    # Save the data of each vehicle in a separate csv-file
    header = ['SIMSEC', 'LinkNo', 'Lane', 'POS']
    for veh in data_per_veh.keys():
        overview_folder = save_loc
        file_name = overview_folder + str(int(veh)) + '.csv'
        with open(file_name, 'a+', newline='') as file:
            spamwriter = csv.DictWriter(file, fieldnames=header)
            if file.tell() == 0:
                spamwriter.writeheader()
            for k in range(len(data_per_veh[veh]['SIMSEC'])):
                spamwriter.writerow({key: data_per_veh[veh][key][k] for key in data_per_veh[veh].keys()})
    return

def trajectory_files(location, save_loc,save = 0):
    # Prepare the full data set in order to save it in a file per vehicle (uncomment last line)
    # or in order to use further analysis

    # MEANING OF THE KEYS
    # SIMSEC = simulation second of recording
    # lanelinkno = number of lane (1-69)
    # lane index = index of the lane
    # pos = position in meters from begin of the lane
    # poslat = lateral position from right of the lane

    data_per_veh = {}
    i = 0
    for chunk in pd.read_csv(location, sep = ';',chunksize=1500000,skiprows=18): # read in chunks --> easier for memory
        chunk = chunk.values.tolist()
        for k in chunk:
            if k[1] not in data_per_veh.keys():
                data_per_veh[k[1]] = {}
                data_per_veh[k[1]]['SIMSEC'] = []
                data_per_veh[k[1]]['LinkNo'] = []
                data_per_veh[k[1]]['Lane'] = []
                data_per_veh[k[1]]['POS'] = []
            data_per_veh[k[1]]['SIMSEC'].append(k[0])
            data_per_veh[k[1]]['LinkNo'].append(k[2])
            data_per_veh[k[1]]['Lane'].append(k[3])
            data_per_veh[k[1]]['POS'].append(k[4])

        if save == 1: # save each vehicle into separate file
            save_files(data_per_veh, save_loc)
        data_per_veh.clear()
        i += 1
        print('Saving chunk: ', i)
    return data_per_veh

def get_network(network_file):
    # Read the file with the network characteristics of the links:
    # the file contains information such as link length, connector y/n, from link, to link etc
    # OUTPUT: dictionary with keys LinkNumber and values:
    #           '$LINK:NO': link number
    #           'NAME': name of the link
    #           'LINKBEHAVTYPE': type of link (urban (=1.0), motorway...)
    #           'DISPLAYTYPE':
    #           'LEVEL':
    #           'NUMLANES':
    #           'LENGTH2D': Length of the lane in meters
    #           'ISCONN': 1 if the link is a connector, 0 if normal link
    #           'FROMLINK': if link is connector, from what lane does it start
    #           'TOLINK': if link is connector, to what lane does it go
    #           'LNCHGDISTDISTDEF':
    #           'HASOVTLN':

    workbook = xlrd.open_workbook(network_file)
    worksheet = workbook.sheet_by_index(0)
    first_row = [] # The row where we stock the name of the column
    for col in range(worksheet.ncols):
        first_row.append( worksheet.cell_value(0,col) )
    # transform the workbook to a list of dictionnary
    data ={}
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]]=worksheet.cell_value(row,col)
        data[elm['$LINK:NO']] = elm
    return data


def find_path(data):
    # add a key that has the total distance the car has travelled
    vehicle_path = data
    vehicle_path['TOTPOS'] = []
    for k in range(len(data['SIMSEC'])):
        if k == 0: # first point in the dataset
            vehicle_path['TOTPOS'].append(vehicle_path['POS'][k]) # distance travelled on the first link before the first record
        elif data['LinkNo'][k] == data['LinkNo'][k-1]: # no change of travelled link
            vehicle_path['TOTPOS'].append(vehicle_path['TOTPOS'][k-1] + data['POS'][k] - data['POS'][k-1])
        else:
            vehicle_path['TOTPOS'].append(vehicle_path['TOTPOS'][k-1] + data['POS'][k])
    return vehicle_path

def link_on_path(data,link):
    # return 1 if link is on path of the vehicle
    if link in data['LinkNo']:
        return 1
    return 0

def get_distance_network(data, network):
    # Find the entire distance of the links on the network that lie on the path of the vehicle
    path = [] # all links traveled by the vehicle
    for k in range(len(data['LinkNo'])):
        if data['LinkNo'][k] not in path:
            path.append(data['LinkNo'][k])
        else:
            pass
    distance = {}
    for i in range(len(path)): #iteration over links in path
        if i == 0: #first link on the path
            distance[path[i]] = network[path[i]]['LENGTH2D'] #lenght of the link
        else:
            distance[path[i]] = distance[path[i-1]] + network[path[i]]['LENGTH2D']
    return distance

def read_trafficlights(TL_file):
    # Read traffic light states:
    # OUTPUT: dictionary 'signal' with 2 keys:
    #   - 'State': dictionary with:
    #               keys: (SignalController, SignalGroup)
    #               values: dictionary with
    #                       keys: 'SIMSEC', 'CYCLETIME', 'SC','Sgroup','NewState','DurationOfLastState', 'SCType','Cause'
    #                       values: list containing all changes over the timespan
    #   - 'Config': dictionary with:
    #               keys: Link
    #               values: dictionary with
    #                       keys: 'SC', 'Sgroup, 'Link', 'Lane', 'At'
    #                       values: float-value from file
    workbook = xlrd.open_workbook(TL_file)
    worksheet = workbook.sheet_by_name('TLstatus')
    first_row = [] # The row where we stock the name of the column
    for col in range(worksheet.ncols):
        first_row.append( worksheet.cell_value(0,col) )
    signal = {}
    data ={}
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]]=worksheet.cell_value(row,col)
        if (int(elm['SC']), int(elm['Sgroup'])) not in data.keys():
            data[(int(elm['SC']),int(elm['Sgroup']))] = {}
        for key in elm.keys():
            if key not in data[(int(elm['SC']),int(elm['Sgroup']))].keys():
                data[(int(elm['SC']),int(elm['Sgroup']))][key] = []
            data[(int(elm['SC']),int(elm['Sgroup']))][key].append(elm[key])
    signal['State']= data

    worksheet2 = workbook.sheet_by_name('TLconfig')
    first_row2 = [] # The row where we stock the name of the column
    for col in range(worksheet2.ncols):
        first_row2.append( worksheet2.cell_value(0,col))
    config = {}
    for row in range(1,worksheet2.nrows):
        elm = {}
        for col in range(worksheet2.ncols):
            elm[first_row2[col]] = worksheet2.cell_value(row,col)
        if elm['Link'] not in config.keys():
            config[elm['Link']] = {}
        config[elm['Link']][elm['Lane']] = elm
        #print(config[elm['Link']][elm['Lane']])
    signal['Config'] = config
    return signal


#%% the next part does not further play a role in the model but is for plotting and analysing
# Updated versions that are easier to use and have better functionalities can be found in
# trajectory_data.py

def plot_path(data,start, end, TL_file):
    # plot the trajectories in an x-t plot
    veh_on_path = []
    data2 = data
    maximal_time = 0
    plt.figure(dpi=150)
    path = {}
    pltmin = 55750
    pltmax = 56250
    for veh in data2.keys():
        if int(data2[veh]['LinkNo'][0]) == start and int(data2[veh]['LinkNo'][-1]) == end: # first and last link of the path
            df = pd.DataFrame.from_dict(data2[veh])
            Lcolors = np.array([int(df.Speed[k]*3.6) for k in range(len(df.Speed))])
            plt.scatter(df.SIMSEC,df.TOTPOS,c=Lcolors, cmap= 'RdYlGn', marker ='s', s=.5)
            veh_on_path.append(veh)
            if max(data2[veh]['SIMSEC']) > maximal_time:
                maximal_time = max(data2[veh]['SIMSEC'])
            for k in range(len(data2[veh]['LinkNo'])):
                path[data2[veh]['LinkNo'][k]] = str(int(data2[veh]['Lane'][k]))
    #cbar = plt.colorbar()
    #cbar.set_label('Speed [km/h]')
    if len(veh_on_path) > 0:
        distance = get_distance_network(data2[veh_on_path[0]],get_network())
    else:
        distance = {}
    TL = read_trafficlights(TL_file)
    for link in distance.keys(): # problem: cars not going straight: additional length of link straight counted as well as length of turning lane
        if link in TL['Config'].keys() and path[link] in TL['Config'][link].keys():
            sg = (int(TL['Config'][link][path[link]]['SC']),int(TL['Config'][link][path[link]]['Sgroup']))
            for k in range(len(TL['State'][sg]['SIMSEC'])-1):
                if TL['State'][sg]['NewState'][k] == ' red       ':
                    col = 'tab:red'
                elif TL['State'][sg]['NewState'][k] == ' amber     ':
                    col = 'tab:orange'
                else:
                    col = 'tab:green'
                xmin = int(float(TL['State'][sg]['SIMSEC'][k].strip()))
                xmax = int(float(TL['State'][sg]['SIMSEC'][k].strip())+int(float(TL['State'][sg]['DurationOfLastState'][k+1])))
                if xmin >= pltmin and xmax <= pltmax:
                    plt.axhline(y = distance[link], xmin= xmin/1000, xmax= xmax/1000, color = col,ls= '-', lw=2)#, linestyles='solid')#, lw =5)
    plt.xlim(55750,56250)
    plt.title(str('xt-plot of cars on path from link '+ str(start) + ' to link ' +str(end)))
    plt.xlabel('Time [sec]')
    plt.ylabel('Travelled distance [m]')
    plt.show()
    return

def plot_link(data, link, TL_file):
    # Plot xt-diagram per link
    TL = read_trafficlights(TL_file)
    if link in TL['Config'].keys():
        print('link = ', link)
        for lane in TL['Config'][link].keys():
            print('lane = ', lane)
            for veh in data.keys():
                if link in data[veh]['LinkNo']:
                    df = pd.DataFrame.from_dict((data[veh]))
                    df = df.loc[(df['LinkNo'] == float(link)) & (df['Lane'] == float(lane))]
                    if not df.empty:
                        Lcolors = np.array([int(k*3.6) for k in df.Speed])
                        plt.scatter(df['SIMSEC'], df['POS'],c=Lcolors, cmap= 'RdYlGn',marker = 's', s=.5)

            sg = (int(TL['Config'][link][lane]['SC']),int(TL['Config'][link][lane]['Sgroup'])) #Take into account lane here !!!!!
            for k in range(len(TL['State'][sg]['SIMSEC'])-1):
                if TL['State'][sg]['NewState'][k] == ' red       ':
                    col = 'tab:red'
                elif TL['State'][sg]['NewState'][k] == ' amber     ':
                    col = 'tab:orange'
                else:
                    col = 'tab:green'
                xmin = int(float(TL['State'][sg]['SIMSEC'][k]))
                xmax = int(float(TL['State'][sg]['SIMSEC'][k]) + int(float(TL['State'][sg]['DurationOfLastState'][k + 1])))

                plt.axhline(y=TL['Config'][link][lane]['At'],xmin=xmin/12600,xmax=xmax/12600, color=col,ls='-', lw=4)

            plt.title('x-t plot of link '+ str(link) + ' & lane: ' + str(lane))
            plt.xlabel('Time [sec]')
            plt.ylabel('Travelled distance [m]')
            plt.show()
    return

def get_vehID(file):
    # Turn filename into only vehID
    vehID = str(file)
    for k in range(len(file)-5):
        if vehID[k:k+5] == 'vehID':
            vehID = vehID[k:-4]
    return vehID

def get_speed(data):
    #calculate instantaneous speed
    speed = []
    if len(data['SIMSEC']) >1:
        for k in range(len(data['SIMSEC'])):
            speedveh = (data['TOTPOS'][k] - data['TOTPOS'][k-1]) / (data['SIMSEC'][k]-data['SIMSEC'][k-1])
            speed.append(speedveh)
    else:
        speed.append(0)
    return speed


def get_vehicle(save_map, network_file,TL_file, penetration_rate=1):
    # call for plot + make turning fractions and total flows on the links

    #Read files per vehicle & network data
    Overview_folder = save_map
    csvfiles = glob.glob(os.path.join(Overview_folder,'*csv'))
    network = get_network(network_file)
    # Initialize
    flow_counter = {link: 0 for link in network.keys()}
    vehicle_path = {}
    # Select limited number of trajectories based on the penetration rate
    penetration = int(penetration_rate * len(csvfiles))
    stop = len(csvfiles)
    if penetration_rate < 1:
        list_pen_rate = random.sample(range(0,stop),penetration)
    else:
        list_pen_rate = [i for i in range(len(csvfiles))]

    for file in csvfiles:
        vehID = get_vehID(file)
        for k in range(len(list_pen_rate)):
            if list_pen_rate[k] == int(vehID[5:]):
                data = {}
                with open(file,'r') as file: # Read vehicle files
                    reader = csv.DictReader(file)
                    for row in reader:
                        dt = dict(row)
                        for key in dt.keys():
                            if key not in data.keys():
                                data[key] = []
                            data[key].append(float(row[key]))
                path = find_path(data) # list with links on the path
                vehicle_path[vehID] = path
                data['Speed'] = get_speed(path)
                for link in network.keys():
                    flow_counter[link] += link_on_path(data, link)
            else:
                pass
        print(vehID)
    print('Plotting')
    plot_path(vehicle_path, 12.0, 1.0, TL_file)
    plot_link(vehicle_path, 12.0, TL_file)

    print(flow_counter)
    turning_fractions = {}
    for link in network.keys():
        if network[link]['ISCONN'] == 1:
            from_link = network[link]['FROMLINK']
            to_link = network[link]['TOLINK']
            if from_link not in turning_fractions.keys():
                turning_fractions[from_link] = {}
            if flow_counter[link] != 0:
                turning_fractions[from_link][to_link] = flow_counter[link]/flow_counter[from_link]
    return turning_fractions

# save_map = 'Data\Simulation\Shockwaves\paths3'
# network_file = 'Data\Simulation\\network.xlsx'
# TL_file = 'Data\Simulation\Shockwaves\Deventer_TL.xlsx'
# flow_counter = get_vehicle(save_map,network_file, TL_file)
#flow_counter2 = get_vehicle(0.01)

#print(flow_counter)
