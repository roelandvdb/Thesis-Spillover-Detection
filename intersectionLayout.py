from xml.dom import minidom
import xml.etree.ElementTree as ET
import geopandas
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'


def read_network(): # shapefile of the basemap of the network
    segments_file = 'Data\VLOG\Shapefile_bb\segments_clipped.shp'
    segments = geopandas.read_file(segments_file)
    return segments
segments = read_network()
def create_intersection(tree): # geodataframe of the intersection based on ITF-file
    dict_intersection = {'name': [], 'latitude': [], 'longitude': []}
    root = tree.getroot()
    dict_intersection['name'].append(root[2][1][0][0].text)
    dict_intersection['latitude'].append(int(root[2][1][0][3][0].text)/10**7)
    dict_intersection['longitude'].append(int(root[2][1][0][3][1].text)/10**7)
    df = pd.DataFrame.from_dict(dict_intersection, orient='index').transpose()
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
    return gdf

def create_sensors(tree): #geodataframe with the locations of the detectors on the intersection
    dict_sensors = {'ID': [],'name': [], 'type': [], 'latitude': [], 'longitude': [], 'Length': [], 'laneID': [], 'distance': []}
    root = tree.getroot()
    for sensor in root.iter('sensor'):
        dict_sensors['ID'].append(sensor[0].text)
        dict_sensors['name'].append(sensor[2].text)
        dict_sensors['type'].append(sensor[3].text)
        dict_sensors['latitude'].append(int(sensor[6][0].text)/10**7)
        dict_sensors['longitude'].append(int(sensor[6][1].text)/10**7)

        if sensor[3].text == 'inductionLoop':
            dict_sensors['laneID'].append(sensor[9][0][0].text)
            dict_sensors['distance'].append(sensor[9][0][1].text)
            dict_sensors['Length'].append(float(sensor[7].text) / 100)
            #print(sensor[7].text)
            #print(float(sensor[7].text)/100)
        else:
            dict_sensors['laneID'].append('None')
            dict_sensors['distance'].append('None')
            dict_sensors['Length'].append('None')
    dfsensor = pd.DataFrame.from_dict(dict_sensors, orient='index').transpose()
    gdfsensor = geopandas.GeoDataFrame(dfsensor,
                                       geometry=geopandas.points_from_xy(dfsensor.longitude, dfsensor.latitude))
    return gdfsensor

def create_lanes(tree, intersection):
    # GDF with the lane data, as well as a visual matching to the segment in the base map
    lanes = {'laneID': [], 'name': [], 'use': [], 'stopline_lat': [], 'stopline_lon': [], 'connectingLane': [],
        'signalGroup': [], 'segments': [], 'ingress': []}
    root = tree.getroot()
    for lane in root.iter('genericLane'):
        children = lane.getchildren()
        #print(children)
        ch = []
        for child in children:
            ch.append(child.tag)
        lanes["laneID"].append(lane[0].text)
        lanes['name'].append(lane[1].text)
        if lane[3][2].attrib == 'bikeLane':
            lanes['use'].append('bike')
        else:
            lanes['use'].append('car')
        used = 0
        for node in lane.iter('nodeXY'):
            for nodeAttributeXY in node.iter('nodeAttributeXY'):
                if nodeAttributeXY.text == 'stopLine':
                    lanes['stopline_lat'].append(int(node[0][0].text) / 10 ** 7)
                    lanes['stopline_lon'].append(int(node[0][1].text) / 10 ** 7)
                    used = 1
        if used == 0:
            lanes['stopline_lat'].append(None)
            lanes['stopline_lon'].append(None)

        added = 0
        for connection in lane.iter('connection'):
            if added == 0:
                lanes['connectingLane'].append(connection[0][0].text)
                lanes['signalGroup'].append(connection[1].text)
                added = 1
        if 'connectsTo' not in ch:
            lanes['connectingLane'].append(None)
            lanes['signalGroup'].append(None)

    for i in range(len(lanes['laneID'])):
        if intersection == 'A010':
            if lanes['laneID'][i] == '1':
                lanes['segments'].append((3510685, 3510684)) # coinciding segments, determined visually (automated = outside scope)
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '2':
                lanes['segments'].append((3510685, 3510684))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '3':
                lanes['segments'].append((3510409, 3510410, 3510411))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '4':
                lanes['segments'].append((3510655,3510654,3510653, 3510670,3510669, 15513687))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '5':
                lanes['segments'].append((3510682, 3510683))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '6':
                lanes['segments'].append((3510655,3510654,3510653,3510667,3510668,3510677,3510679))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '7':
                lanes['segments'].append((3510655,3510654,3510653,3510667,3510668,3510677,3510679))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '12':
                lanes['segments'].append((3509343, 3509342,3509341,3509340))
                lanes['ingress'].append(1)
            else:
                lanes['segments'].append(())
                lanes['ingress'].append((2))
        elif intersection == 'A020':
            if lanes['laneID'][i] == '1':
                lanes['segments'].append((7955984, 7955985, 7955986))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '2':
                lanes['segments'].append((7955984, 7955985, 7955986, 16828836, 16828838, 18733050, 18733051))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '3':
                lanes['segments'].append((7955984, 7955985, 7955986))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '4':
                lanes['segments'].append((7955978, 7955977, 7955976))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '5':
                lanes['segments'].append((7955978, 7955977, 7955976))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '6':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '7':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '8':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '13':
                lanes['segments'].append((12021447, 18076487, 3510674,3510675,3510676))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '14':
                lanes['segments'].append((12021447, 18076487, 3510674, 3510675, 3510676))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '16':
                lanes['segments'].append((11139463, 11139464,11139465,11139466))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '17':
                lanes['segments'].append((7955979,7955980,7955981,7955982))
                lanes['ingress'].append(0)
            else:
                lanes['segments'].append(())
                lanes['ingress'].append(2)
        elif intersection == 'A023':
            if lanes['laneID'][i] == '1':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '2':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '3':
                lanes['segments'].append((7955978, 7955977, 7955976))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '4':
                lanes['segments'].append((7955978, 7955977, 7955976))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '5':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '6':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '7':
                lanes['segments'].append((3509972, 3509971, 7955974, 7955975))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '8':
                lanes['segments'].append((3509972, 3509971, 7955974, 7955975))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '11':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '12':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '13':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '14':
                lanes['segments'].append((7955972, 7955973))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '15':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '16':
                lanes['segments'].append((3510671, 3509258, 3509259))
                lanes['ingress'].append(1)
            else:
                lanes['segments'].append(())
                lanes['ingress'].append(2)

        elif intersection == 'A026':
            if lanes['laneID'][i] == '1':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '2':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '3':
                lanes['segments'].append((3506670, 3506669, 3506668, 3506667, 3506658))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '4':
                lanes['segments'].append((3506659, 3506660))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '5':
                lanes['segments'].append((3506659, 3506660))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '6':
                lanes['segments'].append((3506769, 3506765))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '7':
                lanes['segments'].append((3506769, 3506765))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '8':
                lanes['segments'].append((3506666,3506665,3509223,3509222, 3509221))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '9':
                lanes['segments'].append((3506666, 3506665, 3509223, 3509222, 3509221))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '10':
                lanes['segments'].append((3509260,3509261,3509262,3509263,3509264))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '11':
                lanes['segments'].append((3509260,3509261,3509262,3509263,3509264))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '12':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '13':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '14':
                lanes['segments'].append((3509252, 3509253,3509254,3509255))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '15':
                lanes['segments'].append((3506666,3506665,3509223,3509222, 3509221))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '16':
                lanes['segments'].append((3506666,3506665,3509223,3509222, 3509221))
                lanes['ingress'].append(1)
            elif lanes['laneID'][i] == '17':
                lanes['segments'].append((3506659, 3506660))
                lanes['ingress'].append(0)
            elif lanes['laneID'][i] == '18':
                lanes['segments'].append((3506769, 3506765))
                lanes['ingress'].append(0)
            else:
                lanes['segments'].append(())
                lanes['ingress'].append(2)
        else:
            lanes['segments'].append(())
    f = {k: len(lanes[k]) for k in lanes.keys()}
    df_lanes = pd.DataFrame.from_dict(lanes, orient='index').transpose()
    return df_lanes

def create_signalgroups(tree): # datafram with the signal groups and coinciding name in the VLOG
    dict_sg = {'name': [], 'signalGroup': [], 'vlogIdx': []}
    root = tree.getroot()
    for sg in root.iter('sg'):
        dict_sg['name'].append(sg[0].text)
        dict_sg['signalGroup'].append(sg[1].text)
        dict_sg['vlogIdx'].append(sg[3].text)
    df = pd.DataFrame.from_dict(dict_sg, orient='index').transpose()
    return df

def plots(result): #create plot of the network + the sensors.
    segments = read_network()
    frames = [result[i]['intersection_layout'] for i in result.keys()]
    gdf = pd.concat(frames)
    frames_sensor = [result[i]['sensors'] for i in result.keys()]
    gdfsensor = pd.concat(frames_sensor)
    px.set_mapbox_access_token(gdf)
    fig = go.Figure()
    import shapely.geometry
    import numpy as np
    lats = []
    lons = []
    names = []
    for feature, name in zip(segments.geometry, segments.linkID):
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
    fig.add_trace(go.Scattergeo(lon=gdf.geometry.x, lat=gdf.geometry.y, hoverinfo='text', text=gdf.name))
    fig.add_trace(
        go.Scattergeo(lon=gdfsensor.geometry.x, lat=gdfsensor.geometry.y, hoverinfo='text', text=gdfsensor.name))
    fig.show()

    fig, ax = plt.subplots(1, 1)
    segments.plot(ax=ax, color='grey', zorder=0)
    gdf.plot(ax=ax, color='black', zorder=5)
    gdfsensor.plot(ax=ax, column='type', legend=True, markersize=3, zorder=10,
                   cmap='brg')  # color='blue', markersize = 1)
    minx, miny, maxx, maxy = gdfsensor.total_bounds
    r = 0.0001
    ax.set_xlim(minx - r, maxx + r)
    ax.set_ylim(miny - r, maxy + r)
    plt.axis('off')
    plt.show()

def read_xml():  # create dictionary containing the aforementioned (geo-)dataframes
    document_folder = 'Data\VLOG\ITF'
    xml_files = glob.glob(os.path.join(document_folder, '*.xml'))

    #INITIALIZE
    intersections = []
    gdf_intersections = {}
    gdf_sensors = {}
    df_lanes = {}
    df_signalgroups = {}

    #EXTRACT INFO FROM ITF
    for document in xml_files:
        tree = ET.parse(document)
        root = tree.getroot()

        intersections.append(root[2][1][0][0].text[:-2])
        gdf_intersections[intersections[-1]] = create_intersection(tree)
        gdf_sensors[intersections[-1]] = create_sensors(tree)
        df_lanes[intersections[-1]] = create_lanes(tree, intersections[-1])
        df_signalgroups[intersections[-1]] = create_signalgroups(tree)
    result = {intersection: {'intersection_layout':gdf_intersections[intersection], 'sensors':gdf_sensors[intersection],
                            'lanes': df_lanes[intersection], 'signalgroups': df_signalgroups[intersection]} for intersection in intersections}

    # If wanted: PLOT INFO FROM RESULTS
    #plots(result)

    return result

# To run example:
# layout = read_xml()
