# make the markov chain
import simulation_data as SD
import csv



def get_trajectories(link, lane, csvfiles):
    # read and store the trajectories that use a certain link and lane + add instantaneous speed
    alldata = {}
    counter = 0
    for file in csvfiles:
        counter +=1
        vehID = SD.get_vehID(file)
        data = {}
        with open(file,'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                dt = dict(row)
                for key in dt.keys():
                    if key not in data.keys():
                        data[key] = []
                    data[key].append(float(row[key]))
        cont = 0
        kaas = []
        for k in range(len(data['SIMSEC'])):
            if link == data['LinkNo'][k]:
                kaas.append(k)
        if len(kaas)>0 and data['Lane'][max(kaas)] == lane:
            cont = 1
        if cont == 1:
            data = SD.find_path(data)
            speeds = SD.get_speed(data)
            data['Speed'] = [round(speeds[k]*3.6,2) for k in range(len(speeds))]
            alldata[vehID] = data
        if counter%1000 == 0:
            print('Already ', counter, 'vehicles loaded!')

    print('Trajectories on link ', link, ' loaded')
    return alldata

def get_cycles2(min_time, max_time):
    #for fixed time steps == 1minute; other methods are possible, but are not included in this script as there was a bug present
    # and they are not further used.
    cycles = []
    for i in range(min_time, max_time, 60):
        cycles.append(i)
    return cycles



