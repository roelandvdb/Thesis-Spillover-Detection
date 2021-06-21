import pyvlog.converters as cv
import pyvlog.parsers as ps
import glob
import os
import pandas as pd
#from datetime import datetime
import datetime
import intersectionLayout as IL
import math
import statistics
import matplotlib.pyplot as plt
import time
import csv


def read_config():
    # This function reads the configuration files for each intersection, indicating the names of the approaches
    config_folder = 'Data\VLOG\Config'
    config_files = glob.glob(os.path.join(config_folder, '*.vlt'))
    intersections = []
    cf = {}
    for file in config_files:
        intersections.append(str(file)[-8:-4])
        intersection = intersections[-1]
        cf[intersection] = []
        f = open(file, 'r')
        for x in f:
            if x != '\n':
                cf[intersection].append(x[:-2])
        f.close()
    return cf


def get_light_cycles(jf, layout):
    # This function reads the traffic signal cycles, based on vlog data and the layout of the intersection, and
    # saves the signal changes for every singal group of the intersection in a dataframe
    signals = {}
    for day in jf.keys():
        signals[day] = {}
        for intersection in jf[day].keys():
            signals[day][intersection] = {}
            all_lanes = layout[intersection]['lanes']['laneID'].tolist()
            all_lanes = [lane for lane in all_lanes if lane is not None]
            #print(all_lanes)
            for lane in all_lanes:
                s = {}
                if len(jf[day][intersection]) == 0:
                    pass
                else:
                    signals[day][intersection][lane] = {}
                    this_lane = layout[intersection]['lanes'][layout[intersection]['lanes']['laneID'] == lane]
                    #print(this_lane)
                    s['lane'] = lane
                    s['timestamp'] = jf[day][intersection]['timestamp'].values.tolist()
                    group = this_lane['signalGroup']
                    group = group.to_numpy()
                    if group[0] is not None:
                        new_df = layout[intersection]['signalgroups'][layout[intersection]['signalgroups']['signalGroup'] == group[0]]
                        signalgroup = 'externeSignaalgroep_' + str(new_df['vlogIdx'].to_numpy()[0])
                        s['signal'] = jf[day][intersection][signalgroup].values.tolist()
                    else:
                        s['signal'] = [None]*len(jf[day][intersection]['timestamp'])
                    s['connectingLane'] = [this_lane['connectingLane'].to_numpy()[0]]*len(jf[day][intersection][signalgroup])
                    s['stopLine_lon'] = [this_lane['stopline_lon'].to_numpy()[0]]*len(jf[day][intersection][signalgroup])
                    s['stopLine_lat'] = [this_lane['stopline_lat'].to_numpy()[0]]*len(jf[day][intersection][signalgroup])
                    signals[day][intersection][lane] = pd.DataFrame.from_dict(s)
    return signals

def get_VLOG(start, end,date_format):
    # Get the measurements of the loop detectors between the start and end time, which can span multiple days.
    # The VLOG is saved in files of 5 minutes, but small deviations per file might occur, where the first timestamp is
    # e.g. 00h01.

    VLOG_folder = 'Data\VLOG\VLOG'
    VLOG_folder_folders = glob.glob(VLOG_folder + '/*/*/')
    jf = {}
    for folder in VLOG_folder_folders:
        start_date = datetime.datetime.strptime(start, date_format)
        end_date = datetime.datetime.strptime(end, date_format)
        start_date = start_date
        end_date = end_date
        day = str(folder)[-5:-1]
        intersection = str(folder)[-10:-6]

        if day not in jf.keys():
            jf[day] = {}
        if intersection not in jf[day].keys():
            jf[day][intersection] = {}
        vlog_files = glob.glob(os.path.join(folder,'*.vlog'))
        for file in vlog_files:
            T = datetime.datetime.strptime(str(file)[-20:-5], '%Y%m%d.%H%M%S')
            if (T >= start_date - datetime.timedelta(minutes = 4)) & (T < end_date):
                if len(jf[day][intersection]) == 0:
                    jf[day][intersection] = cv.file_to_dataframe(file) #Read using pyvlog package
                else:
                    frames = [jf[day][intersection], cv.file_to_dataframe(file)]
                    jf[day][intersection] = pd.concat(frames)
    to_del = []
    for k in jf.keys(): #date
        for w in jf[k].keys(): #intersection
            if len(jf[k][w]) > 0:
                jf[k][w] = jf[k][w][(jf[k][w].timestamp >=  start_date) & (jf[k][w].timestamp < end_date )]
            else:
                to_del.append((k,w)) # delete those that contain data from before the start time, and after the end time
    for elem in to_del:
        del jf[elem[0]][elem[1]]
    return jf

def get_occupancy(vlog, dtime= 60):
    # Calculate the relative occupancy over every detector based on the LD measurements.
    # dtime is the aggregation period used in seconds, and is standard 1 minute.
    occupancy = {}
    for date in vlog.keys():
        occupancy[date] = {}
        for intersection in vlog[date].keys():
            occupancy[date][intersection] = {}
            v = vlog[date][intersection]
            if len(v) > 0:
                start_time = sorted(v.timestamp.to_list())[0]
                end_time = sorted(v.timestamp.to_list())[-1]
                delta = math.floor((end_time - start_time).total_seconds())
                for i in range(0,delta,dtime):
                    vlog_time = v[(v.timestamp >= start_time + datetime.timedelta(seconds=i)) &
                                   (v.timestamp < start_time + datetime.timedelta(seconds=i+dtime))]
                    col = vlog_time.columns.tolist()
                    col = [elem[:-6] for elem in col if 'bezet' in elem]
                    for detector in col:
                        if detector not in occupancy[date][intersection].keys():
                            occupancy[date][intersection][detector] = {}
                        occupancy[date][intersection][detector][str(start_time+datetime.timedelta(seconds=i))] = 1
                        list_detection = vlog_time[detector + '_bezet'].tolist()
                        list_timestamp = vlog_time['timestamp'].tolist()
                        start = start_time+datetime.timedelta(seconds=i)
                        sum = 0
                        if len(list_detection) > 0:
                            blocked = list_detection[0]
                        else:
                            blocked = 0
                        for k in range(len(list_detection)-1):
                            if (list_detection[k] == 1) and (blocked == 0): # first congestion
                                start = list_timestamp[k]
                                blocked = 1
                            elif (list_detection[k] == 0) and (blocked == 1):
                                end = list_timestamp[k]
                                sum += (end-start).total_seconds()
                                blocked = 0
                            elif (list_detection[k] == 1) and (blocked == 1) and k == len(list_detection)-1:
                                end = start_time + datetime.timedelta(seconds=i+dtime)
                                sum += (end-start).total_seconds()
                        occupancy[date][intersection][detector][str(start_time+datetime.timedelta(seconds=i))] = sum/dtime
    return occupancy

def get_flow(vlog,dtime = 60):
    # Calculate flow over the detector, similarly to occupancy
    flow = {}
    for date in vlog.keys():
        flow[date] = {}
        for intersection in vlog[date].keys():
            flow[date][intersection] = {}
            v = vlog[date][intersection]
            if len(v) > 0:
                start_time = sorted(v.timestamp.to_list())[0]
                end_time = sorted(v.timestamp.to_list())[-1]
                delta = math.floor((end_time - start_time).total_seconds())
                for i in range(0,delta,dtime):
                    vlog_time = v[(v.timestamp >= start_time + datetime.timedelta(seconds=i)) &
                                   (v.timestamp < start_time + datetime.timedelta(seconds=i+dtime))]
                    col = vlog_time.columns.tolist()
                    col = [elem[:-6] for elem in col if 'bezet' in elem]
                    for detector in col:
                        if detector not in flow[date][intersection].keys():
                            flow[date][intersection][detector] = {}

                        list_detection = vlog_time[detector + '_bezet'].tolist()
                        if len(list_detection ) > 0:
                            count = list_detection[0]
                            blocked = list_detection[0]
                        else:
                            count = 0
                            blocked = 0
                        for k in range(len(list_detection)-1):
                            if (blocked == 0) and (list_detection[k] == 1): # count every time of start detection
                                count += 1
                                blocked = 1
                            elif (blocked == 1) and list_detection[k] == 0: # don't count end of detection
                                blocked = 0
                        flow[date][intersection][detector][str(start_time + datetime.timedelta(seconds=i))] = count*3600/dtime # vehicle per hour

    return flow

def get_speed(vlog, dtime=60):
    # calculate speed of the vehicles over the detector, similarly to before.
    # Assumes average vehicle length of 4.34 m, which was calibrated such that reasonable speeds were obtained.

    speed = {}
    LayOut = IL.read_xml()

    for date in vlog.keys():
        speed[date] = {}
        for intersection in vlog[date].keys():
            speed[date][intersection] = {}
            v = vlog[date][intersection]
            IntLay = LayOut[intersection]['sensors']
            if len(v) > 0:
                start_time = sorted(v.timestamp.to_list())[0]
                end_time = sorted(v.timestamp.to_list())[-1]
                delta = math.floor((end_time - start_time).total_seconds())
                for i in range(0,delta,dtime):
                    vlog_time = v[(v.timestamp >= start_time + datetime.timedelta(seconds=i)) &
                                   (v.timestamp < start_time + datetime.timedelta(seconds=i+dtime))]
                    col = vlog_time.columns.tolist()
                    col = [elem[:-6] for elem in col if 'bezet' in elem]

                    for detector in col:
                        if detector not in speed[date][intersection].keys():
                            speed[date][intersection][detector] = {}
                        det_df = IntLay.loc[IntLay.ID == detector[9:]]
                        speed[date][intersection][detector][str(start_time + datetime.timedelta(seconds=i))] = 0
                        list_detection = vlog_time[detector + '_bezet'].tolist()
                        list_timestamp = vlog_time['timestamp'].tolist()
                        start = start_time + datetime.timedelta(seconds=i)
                        if len(det_df.Length.to_list()) > 0:
                            if det_df.Length.to_list()[0] != 'None':
                                L = 4.34 + float(det_df.Length.to_list()[0]) # was 4.7
                            else:
                                L = 4.34
                        else:
                            L = 4.34
                        indiv_speeds = []
                        if len(list_detection) > 0:
                            blocked = list_detection[0]
                        else:
                            blocked = 0
                        for k in range(len(list_detection)):
                            if (list_detection[k] == 1) and (blocked == 0): # first congestion
                                start = list_timestamp[k]
                                blocked = 1
                            elif (list_detection[k] == 0) and (blocked == 1):
                                end = list_timestamp[k]
                                indiv_speeds.append(L / ((end-start).total_seconds()))
                                blocked = 0
                            elif (list_detection[k] == 1) and (blocked == 1) and k == len(list_detection)-1: # not correct: not entire path
                                end = start_time + datetime.timedelta(seconds=i+dtime)
                                indiv_speeds.append(L*3.6 / ((end - start).total_seconds()))
                        if len(indiv_speeds) > 0:
                            speed[date][intersection][detector][str(start_time + datetime.timedelta(seconds=i))] = statistics.harmonic_mean(indiv_speeds)

    return speed

def plot_FD(occupancy, speed, flow, dates, sensors):
    # plots the Fundamental diagram of the measurements of every detector. The density is estimated based on the
    # fundamental relationship q= ku
    for intersection in occupancy[dates[0]].keys():
        s = sensors[intersection]['sensors']
        for detector in occupancy[dates[0]][intersection].keys():
            IDs = s.ID.tolist()
            if detector[9:] in IDs:
                x = s[(s.laneID != None) & (s.ID == str(detector[9:]))]
                laneID = x.laneID.to_numpy()
                distance = x['distance'].to_numpy()
                lst_occ = []
                lst_flow = []
                lst_speed = []
                for date in dates:
                    if len(occupancy[date][intersection]) > 0:
                        lst_occ.extend(list(occupancy[date][intersection][detector].values()))
                        lst_flow.extend(list(flow[date][intersection][detector].values()))
                        lst_speed.extend(list(speed[date][intersection][detector].values()))
                from operator import truediv
                lst_speed[:] = [x + 0.01 for x in lst_speed]
                lst_dens = list(map(truediv, lst_flow, lst_speed))
                plt.scatter(lst_dens, lst_flow, marker='o')
                plt.xlabel('density [veh/km]')
                plt.ylabel('flow [veh/h]')
                plt.title('density-flow of detector ' + str(detector[9:]) + ' at \n distance ' + str(
                    distance[0]) + ' of intersection ' + intersection + ' on lane ' + str(laneID[0]))
                plt.grid()
                plt.show()
                plt.scatter(lst_flow, lst_speed, marker='o')
                plt.xlabel('speed [km/h]')
                plt.ylabel('flow [veh/h]')
                plt.title('speed-flow of detector ' + str(detector[9:]) + ' at \n distance ' + str(
                    distance[0]) + 'm of intersection ' + intersection + ' on lane ' + str(laneID[0]))
                plt.grid()
                plt.show()
                plt.scatter(lst_dens, lst_speed, marker = 'o')
                plt.xlabel('density [veh/km]')
                plt.ylabel('speed [km/h]')
                plt.title('density-speed of detector ' + str(detector[9:]) + ' at \n distance ' + str(
                    distance[0]) + 'm of intersection ' + intersection + ' on lane ' + str(laneID[0]))
                plt.grid()
                plt.show()
            time.sleep(3) # delete to generate figures at faster pace
    return


def make_data_for_FD_development():
    # this function stores the data from the full month of VLOG-data in a csv-file such that it can be reused later.
    # This data entails the average flow over the detectors per timestep of 15 min, as well as the other parameters
    # required for the fundamental diagram

    # Define start & end time; peak time will result in limiting the data to be analysed to those in the specified period
    start_time = "2020-10-15T16:00:00Z"
    end_time = "2020-11-15T18:00:00Z"
    date_format = '%Y-%m-%dT%H:%M:%SZ'
    s_date = datetime.datetime.strptime(start_time, date_format).date()
    e_date = datetime.datetime.strptime(end_time, date_format).date()
    peak_start = "T00:00:00Z"
    peak_end = "T23:59:59Z"
    delta = e_date - s_date

    def daterange(date1, date2):
        for n in range(int((date2-date1).days)+1):
            yield date1 + datetime.timedelta(n)
    dates = []
    dates_full = []
    for dt in daterange(s_date, e_date):
        st = dt.strftime(date_format)
        dates.append(st[5:7] + st[8:10])
        dates_full.append(st[:10])
    print(dates_full)

    import time
    begin = time.time()

    # Generate data
    for x in dates_full:
        print(x)
        start = x+peak_start
        end = x + peak_end
        jf = get_VLOG(start,end,date_format)
        if 'flow' not in globals():
            flow = get_flow(jf,900)
        if 'speed' not in globals():
            speed = get_speed(jf, 900)
        if 'occupancy' not in globals():
            occupancy = get_occupancy(jf, 900)
        else:
            flow[str(x[5:7]+x[8:10])] = get_flow(jf,900)[str(x[5:7]+x[8:10])]
            speed[str(x[5:7]+x[8:10])] = get_speed(jf,900)[str(x[5:7]+x[8:10])]
            occupancy[str(x[5:7]+x[8:10])] = get_occupancy(jf,900)[str(x[5:7]+x[8:10])]

    # Store data in new dictionaries for per date, intersection and detector (previously detector not correct detector names)
    new_occupancy = {}
    new_speed = {}
    new_flow = {}
    sensors = IL.read_xml()
    for date in occupancy.keys():
        new_flow[date] = {}
        new_occupancy[date] = {}
        new_speed[date] = {}
        for intersection in occupancy[date].keys():
            new_flow[date][intersection] = {}
            new_occupancy[date][intersection] = {}
            new_speed[date][intersection] = {}
            for detector in occupancy[date][intersection].keys():
                det = sensors[intersection]['sensors'].loc[sensors[intersection]['sensors'].ID == detector[9:]].name.values
                if len(det) > 0:
                    new_occupancy[date][intersection][det[0]] = occupancy[date][intersection][detector]
                    new_speed[date][intersection][det[0]] = speed[date][intersection][detector]
                    new_flow[date][intersection][det[0]] = flow[date][intersection][detector]
    # If wanted: plot the fundamental diagram for every detector !takes a long time
    # plot_FD(occupancy, speed, flow, dates, sensors)

    ExecutionTime = (time.time()- begin)
    print('Execution Time in seconds: ' + str(ExecutionTime))

    # Calculate average flows per 15 min interval
    average_flows = {}
    hours = {intersection:{detector:[] for detector in new_flow[dates[0]][intersection].keys()} for intersection in new_flow[dates[0]].keys()}
    for date in new_flow.keys():
        for intersection in new_flow[date].keys():
            for detector in new_flow[date][intersection].keys():
                for timestamp in new_flow[date][intersection][detector].keys():
                    if timestamp[11:19] not in hours[intersection][detector]:
                        hours[intersection][detector].append(timestamp[11:16])
    for date in new_flow.keys():
        for intersection in new_flow[date].keys():
            average_flows[intersection] = {}
            for detector in new_flow[date][intersection].keys():
                average_flows[intersection][detector] = {}
                for hour in hours[intersection][detector]:
                    average_flows[intersection][detector][hour] = {'list': [], 'average': 0}
                    for timestamp in new_flow[date][intersection][detector].keys():
                        if hour in timestamp:
                            average_flows[intersection][detector][hour]['list'].append(new_flow[date][intersection][detector][timestamp])
                    if len(average_flows[intersection][detector][hour]['list']) > 0:
                        average_flows[intersection][detector][hour]['average'] = statistics.mean(average_flows[intersection][detector][hour]['list'])
                    else:
                        average_flows[intersection][detector][hour]['average'] = 0
    with open('Data\VLOG\Average_OD_New.csv', 'w', newline = '') as f:
        w = csv.writer(f)
        for intersection in average_flows.keys():
            for detector in average_flows[intersection].keys():
                for timestamp in average_flows[intersection][detector].keys():
                    w.writerow([intersection] + [detector] + [timestamp] + [average_flows[intersection][detector][timestamp]])

    # Create csv with flow, occupancy and speed for every timestep in the data
    with open('Data\VLOG\OD_flows.csv', 'w', newline = '') as f:
        w = csv.writer(f)
        for date in new_flow.keys():
            for intersection in new_flow[date].keys():
                for detector in new_flow[date][intersection].keys():
                    for timestamp in new_flow[date][intersection][detector].keys():
                        w.writerow([date] + [intersection] + [detector] + [timestamp] + [new_flow[date][intersection][detector][timestamp]] + [new_speed[date][intersection][detector][timestamp]]
                                   + [new_occupancy[date][intersection][detector][timestamp]])
    return

# To run: make the datafiles required (commented due to reference in other files to this file.)
#make_data_for_FD_development()