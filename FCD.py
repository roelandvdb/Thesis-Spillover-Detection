# This file will transform the FCD-data of the cars into a trajectory with the timestamp and travel time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shapely.geometry
import glob
import os
import geopandas
import pandas as pd
from datetime import datetime
import plotly.io as pio
import VLOG as VLOG
import intersectionLayout as IL
from math import cos, asin, sqrt, pi
import numpy as np
pio.renderers.default = 'browser'
pd.options.display.precision = 9


def get_data(file): # read FCD file to geodataframe
    data = pd.read_csv(file,sep=',', index_col=0, header=0)
    gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.lon, data.lat))
    date_format = '%Y-%m-%dT%H:%M:%SZ'
    list_times = gdf.timestamp.to_list()
    list_epoch = []
    for k in range(len(list_times)):
        list_epoch.append(datetime.strptime(list_times[k], date_format).timestamp())
    gdf['epoch'] = list_epoch
    return gdf

def distance(lat1, lon1, lat2, lon2): #calculate distance from the stopline (+- Euclidean distance)
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) * 1000 #2*R*asin...

def plot_trajectory_plotly(segments, data_per_veh): # plot the position of a single trajectory on the basemap

    fig = go.Figure()
    lats = []
    lons = []
    names = []
    for feature, name in zip(segments.geometry, segments.segmentID):
        if isinstance(feature, shapely.geometry.linestring.LineString):
            linestrings = [feature]
        elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
            linestrings = feature.geoms
        else:
            continue
        for linestring in linestrings:
            x, y = linestring.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [name] * len(y))
            lats = np.append(lats, None)
            lons = np.append(lons, None)
            names = np.append(names, None)
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines', hoverinfo='text', text=names))
    fig.add_trace(
        go.Scattergeo(lon=data_per_veh.geometry.x, lat=data_per_veh.geometry.y, hoverinfo='text',
                      text=data_per_veh.segmentID))
    fig.show()
    return

def plot_trajectory_speed(segments, data_per_veh, veh): # plot trajectory + speed
    fig2, ax = plt.subplots(1, 1)
    segments.plot(ax=ax, color='grey')
    data_per_veh.plot(ax=ax, column=data_per_veh.speed, markersize=10, legend=True,
                      cmap = 'RdYlGn')
    plt.title('VehicleID: ' + veh)
    plt.show()
    return

def matching(fcd, layout, intersection): # match the vehicle to the correct lane
    # fcd: dataframe with key 'segmentID'
    # layout: dictionary: keys: intersection_layout, sensors, lanes, signalgroups
    #       intersection_layout: name latitude, longitude
    #       sensors: name, type, latitude, longitude, laneID, distance
    #       lanes: laneID, name, use, stopline_lat, stopline_lon, connectingLane, signalGroup, segments
    #       signalGroups: name, signalGroup, vlogIdx
    # segments: mapID, segmentID, linkID, beginNodeI, endNodeID, seqNr, optTTMs, cumOptTTMs, lengthMm, cumDistMm, spdLtTTMs,
    #           jmTSpdTTMs, roadClass, optSpeedKP, maxSpeedKP, bearing, centerPoin, name, geometry
    list_segments = fcd.segmentID.tolist()
    list_segments = list(set(list_segments))
    Y = layout[intersection]['lanes']
    Y = Y[Y.laneID.isna() == False]
    lanes_segments = Y['segments'].tolist()
    possible_laneIDs = {}
    for seg in list_segments:
        for x in lanes_segments:
            if x is not None:
                if seg in x:
                    l = Y[Y['segments'] == x]
                    possible_laneIDs[seg] = l.laneID.tolist()
    for s in possible_laneIDs.keys(): # iterate over segments
        if len(possible_laneIDs[s]) == 1:
            pass
        elif len(possible_laneIDs[s]) > 1:
            timestamp_lane = fcd[fcd['segmentID'] == s].epoch.tolist()
            next_segments = fcd[(fcd['epoch'] > max(timestamp_lane)) &
                                 (fcd['epoch'] < max(timestamp_lane) + 120)].segmentID.tolist()
            for r in next_segments:
                if r in possible_laneIDs.keys():
                    next_lanes = possible_laneIDs[r]
                    connecting_lane = Y[(Y.laneID.isin(next_lanes)) & (Y.ingress == 0)]
                    connecting_lane = connecting_lane.laneID.tolist()
                    if len(connecting_lane) >= 1:
                        ingress_lane = Y[(Y.connectingLane.isin(connecting_lane)) & (Y.ingress == 1)]
                        current_lane = ingress_lane.laneID.tolist()
                        for elem in possible_laneIDs[s]:
                            if elem not in current_lane:
                                possible_laneIDs[s].remove(elem)
    return possible_laneIDs

def plot_per_lane(data, layout, signals, date, intersection,start_time,end_time, veh_ids):
    # Plot all vehicles within certain start-stop time
    lane_df = layout[intersection]['lanes'][layout[intersection]['lanes'].laneID == lane]
    data_per_time = data[(data.segmentID.isin(lane_df.segments.tolist()[0]))]
    points_dataxT = data_per_time.geometry.x.tolist()
    points_datayT = data_per_time.geometry.y.tolist()
    point_stopx = signals[date][intersection][lane]['stopLine_lon']
    point_stopy = signals[date][intersection][lane]['stopLine_lat']
    if isinstance(point_stopx[0], float):
        data_per_time['dist'] = [-distance(points_dataxT[i], points_datayT[i], point_stopx[0], point_stopy[0]) for i in
                                 range(len(points_dataxT))]
        col = []
        for x in signals[date][intersection][lane].signal:
            if x == 0:
                col.append('red')
            elif x == 1:
                col.append('green')
            elif x == 2:
                col.append('orange')
            else:
                col.append('grey')
        for vehicle in veh_ids:
            data_veh = data_per_time[data_per_time.sessionID == vehicle].sort_values(by = ['timestamp'])
            plt.plot(data_veh.epoch, data_veh.dist, marker='o')
        plt.scatter(signals[date][intersection][lane]['timestamp'] / 10 ** 9 - 3600,
                    [0] * (len(signals[date][intersection][lane]['timestamp'])), c=col, marker='_')
        plt.xlim(start_time, end_time)
        plt.title('intersection: ' + intersection + ', laneID: ' + lane)
        plt.show()
    return

def create_signaltimings(signals, intersection, lane, date):
    # make list of timestamps with signal changes
    s = signals[date][intersection][lane].sort_values(by = ['timestamp'])
    list_epoch = s.timestamp.tolist()
    list_signal = s.signal.tolist()
    change_list = {}
    change_list['epoch']  = []
    change_list['signal'] = []
    for i in range(2, len(list_epoch)):
        if list_signal[i] == 0 and list_signal[i-1] == 2: #was orange, now red
            change_list['epoch'].append(list_epoch[i])
            change_list['signal'].append(0) # 0 = becomes red
        elif list_signal[i] == 2 and list_signal[i-1] == 1: #was green, now orange
            change_list['epoch'].append(list_epoch[i])
            change_list['signal'].append(2)
        elif list_signal[i] == 1 and list_signal[i-1] == 0: #was red, now green
            change_list['epoch'].append(list_epoch[i])
            change_list['signal'].append(1) # 1 = becomes green
    reduced_signals= pd.DataFrame.from_dict(change_list)
    return reduced_signals

def get_shockwaves(data, signals, intersection, lane, date):
    # Estimate the experienced shockwaves (rough estimate based on assumption that every TL has an effect)

    # select & sort signals and data
    reduced_signals = create_signaltimings(signals, intersection,lane,date)
    data = data.sort_values(by = ['epoch'])

    # define moment when vehicle passes the intersection
    data_passing = data[(data.dist >= -15) & (data.speed >= 15)] #if you are moving and the distance to the intersection is smaller than 15m
    if data_passing.empty:
        time_passing = data.epoch.max()
    else:
        time_passing = data_passing.epoch.min() # time when the vehicle passes the stopline (should maybe become max distance)

    # define moment when vehicle is at standstill
    data_stopped = data[(data.speed <= 5)] # select all data where speed < 5km/h
    distance_list = data_stopped.dist.tolist() # make list of all distances where vehicle stands still
    dlist = []
    for distance in distance_list:
        already_in = 0
        for x in dlist:
            if (distance - 15 <= x <= distance + 15) == True: # if a vehicle is within 15m of a point already in --> already a group
                already_in = 1
        if already_in == 0:
            dlist.append(distance)
    grouped = {}
    for elem in dlist:
        grouped[elem] = data_stopped[(data_stopped.dist <= elem + 15) & (data_stopped.dist >= elem - 15)] # group data when vehicle is stopped
    start_end_points = {key: {} for key in grouped.keys() if len(grouped[key].index) > 3} # only consider stop if it has at least 3 members
    for key in grouped.keys():
        if len(grouped[key].index) > 3: #only consider stop if at least 3 members
            start_end_points[key]['start'] = grouped[key].epoch.min() #start of a stop
            start_end_points[key]['end'] = grouped[key].epoch.max() # end of a stop
            start_end_points[key]['dist'] = grouped[key].dist.mean() # average distance of the stop
    if len(grouped.keys()) >= 1: # if at least one group of cars standing still

        keys = sorted(list(start_end_points.keys()))
        red = reduced_signals
        end_red = red[(red.signal == 1) & (red.epoch*10**(-9) - 3600 < time_passing)] # time of changes from red to green
        end_red = end_red.epoch.tolist()

        end_green = red[(red.signal == 0) & (red.epoch < end_red[-1])] # time of changes from green to red
        end_green = end_green.epoch.tolist()

        for i in range(len(end_red)):
            if i <= len(keys):

                last_red = end_red[-i]*10**(-9) - 3600
                last_green = end_green[-i]*10**(-9) - 3600
                if len(keys) != 0:
                    start_end_points[keys[-i]]['signalstart'] = last_red
                    start_end_points[keys[-i]]['signalend'] = last_green
    return start_end_points

# Folder and files of FCD Data
folder = 'Data\FCD'
files = glob.glob(os.path.join(folder, '*'))

# Folder and files of Segments
segments_file = 'Data\VLOG\Shapefile_bb\segments_clipped.shp'
segments = geopandas.read_file(segments_file)

# Start & end time
date_format = '%Y-%m-%dT%H:%M:%SZ'
start_T = "2020-10-30T16:00:00Z" #!!!! winter hour from 1 november 2020 --> inconsistency not solved yet
start_date = start_T[0:10]
start_time = datetime.strptime(start_T, date_format).timestamp()
end_T = "2020-10-30T17:00:00Z"
end_date = end_T[0:10]
end_time = datetime.strptime(end_T, date_format).timestamp()

# Call VLOG & Intersection Layout for data on signals
layout = IL.read_xml()
vlog = VLOG.get_VLOG(start_T,end_T,date_format)
signals = VLOG.get_light_cycles(vlog,layout)

intersections = list(layout.keys())
laneIDs = {intersection: [] for intersection in intersections}

for key in laneIDs:
    l = layout[key]['lanes']
    l = l[(l.segments != ()) & (l.segments.isna() == False)]
    laneIDs[key] = l.laneID.tolist()


for file in files:
    print(str(file)[-10:])
    if (str(file)[-10:] == start_date) or (str(file)[-10:] == end_date):
        date = str(file)[-5:-3] + str(file)[-2:]

        # Make list of all vehicles recorded in this file
        data = get_data(file)
        data = data[data.segmentID.isna() == False]
        data_sid = data['segmentID'].astype(int)
        data['segmentID'] = data_sid
        data= data[(data.epoch >= start_time) & (data.epoch <= end_time)]

        veh_ids = data['sessionID'].to_list()
        veh_ids = list(set(veh_ids))

        for intersection in intersections:
            intersection = 'A020'
            print(intersection)
            for lane in laneIDs[intersection]:
                # Uncomment to plot the vehicles per lane over the researched time:
                #plot_per_lane(data, layout, signals, date, intersection,start_time, end_time,veh_ids)

                # Print path of all vehicles on a certain segment during start-stop time
                counter = 0
            for veh in veh_ids:

                data_per_veh = data[data.sessionID == veh]
                data_per_veh.sort_values('timestamp')
                veh_lanes = matching(data_per_veh, layout, intersection) # dictionary = {segment: [laneID]}
                for laneID in veh_lanes.values():
                    if len(laneID) > 0:
                        Y = layout[intersection]['lanes']
                        Y = Y[Y['laneID'] == laneID[0]].ingress.tolist()
                        if Y[0] == 1:
                            lane = laneID[0]
                segments_list = layout[intersection]['lanes'][layout[intersection]['lanes'].laneID == lane]
                data_per_veh_intersection = data_per_veh[data_per_veh.segmentID.isin(segments_list.segments.tolist()[0])]
                if len(data_per_veh_intersection) > 0:
                    points_datax = data_per_veh_intersection.geometry.x.tolist()
                    points_datay = data_per_veh_intersection.geometry.y.tolist()
                    point_stopx = signals[date][intersection][lane]['stopLine_lon'][0]
                    point_stopy = signals[date][intersection][lane]['stopLine_lat'][0]
                    print( [point_stopx, point_stopy])

                    if isinstance(point_stopx, float):
                        data_per_veh_intersection['dist'] = [-distance(points_datax[i], points_datay[i], point_stopx, point_stopy) for i
                                                 in range(len(points_datax))]
                        plt.figure()
                        plt.scatter(data_per_veh_intersection.epoch, data_per_veh_intersection.dist)
                        shockwaves = get_shockwaves(data_per_veh_intersection, signals, intersection, lane, date)
                        for key in shockwaves.keys():
                            if 'start' in shockwaves[key].keys():
                                pass
                                plt.scatter(shockwaves[key]['start'], shockwaves[key]['dist'], c = 'red')
                                plt.scatter(shockwaves[key]['end'], shockwaves[key]['dist'], c = 'red')
                                if 'signalstart' in shockwaves[key].keys():
                                    plt.scatter(shockwaves[key]['signalstart'], 0, c = 'green')
                                    plt.plot((shockwaves[key]['signalstart'], shockwaves[key]['end']), (0, float(shockwaves[key]['dist'])))
                                    plt.plot((shockwaves[key]['signalend'], shockwaves[key]['start']), (0, float(shockwaves[key]['dist'])))

                        s = signals[date][intersection][lane]
                        t = data_per_veh_intersection.epoch

                        s = s[(s.timestamp >= (t.min()+3600-50)*10**9) & (s.timestamp <= (t.max()+3600+50)*10**9)]
                        col = []
                        for R in s.signal:
                            if R == 0:
                                col.append('red')
                            elif R == 1:
                                col.append('green')
                            elif R == 2:
                                col.append('orange')
                            else:
                                col.append('grey')

                        plt.scatter(s['timestamp'] / 10 ** 9 - 3600,[0] * (len(s['timestamp'])),c=col, marker = 'o', zorder = 2)
                        plt.plot(s['timestamp'] / 10 ** 9 - 3600,[0] * (len(s['timestamp'])), marker = '_', zorder = 1)
                        plt.title('vehicle ID: ' + veh + '\n' + 'intersection: ' + intersection + ', lane: ' + lane)
                        plt.xlabel('Time in seconds since January 1, 1970 [s]')
                        plt.ylabel('Distance of the trajectory from the stopline [m]')
                        plt.show()



                # Plot speed of the vehicle along its trajectory
                #plot_trajectory_speed(segments, data_per_veh,veh)

                # Plot location and trajectory
                #plot_trajectory_plotly(segments, data_per_veh)

                counter += 1
                print('Counter: ', counter)
                input('press Enter to continue')


