
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import sklearn.linear_model as LM
import numpy as np
from operator import truediv

file = 'Data\VLOG\OD_flows.csv'
columns = ['date','intersection', 'detector', 'timestamp', 'flow', 'speed', 'occupancy']

LD_measurements = pd.read_csv(file, sep = ',', names = columns, header= None)

for IT in set(LD_measurements.intersection.to_list()):
    LD = LD_measurements.loc[LD_measurements.intersection == IT]
    for detector in set(LD.detector.to_list()): #for every detector
        if 'dk' not in detector: # don't consider 'drukknop'-detectors
            print(IT , ' ', detector)
            det  = LD.loc[LD.detector == detector]
            flow = det.flow.to_list()
            speed = det.speed.to_list()
            occ = det.occupancy.to_list()
            speed2 = [x + 0.01 for x in speed]
            dens = list(map(truediv, flow, speed2))
            fig, axes = plt.subplots(2,2)
            axes[1,0].scatter(dens, flow)
            axes[1,0].set_xlabel('Density')
            axes[1,0].set_ylabel('Flow')
            axes[0,0].scatter(dens, speed)
            axes[0,0].set_ylabel('Speed')
            axes[0,0].set_xlabel('Density')
            axes[0,1].scatter(flow,speed)
            axes[0,1].set_ylabel('Speed')
            axes[0,1].set_xlabel('Flow')
            fig.suptitle('Fundamental Diagram of detector ' + detector +' on intersection ' + IT)
            print(len(dens))
            dens = [dens[i] for i in range(len(dens)) if ( (dens[i] != 0) and (not math.isnan(dens[i])))]
            speed = [speed[i] for i in range(len(speed)) if ((speed[i]!=0) and (not math.isnan(speed[i])))]
            flow = [flow[i] for i in range(len(flow)) if ((flow[i]!=0) and (not math.isnan(flow[i])))]

            # fit the Greenshield, Greenberg and Underwood models
            if len(dens) > 0 and len(dens) == len(speed) and len(speed) == len(flow):
                X = np.array(dens).reshape(-1,1)
                y = np.array(speed).reshape(-1,1)
                reg = LM.LinearRegression().fit(X,y)
                y_pred = reg.predict(X)
                print(reg.coef_)
                print(reg.intercept_)
                score = reg.score(X,y)
                print('Greenshield:', score)
                axes[1,0].scatter(X, y_pred, c = 'black', linewidth = 1)

                X = np.array(np.log(dens)).reshape(-1,1)
                y = np.array(speed).reshape(-1,1)
                X_plot = np.array(dens)
                reg2 = LM.LinearRegression().fit(X,y)
                y_pred = reg2.predict(X)
                score = reg2.score(X,y)
                print('Greenberg:', score)
                axes[1, 0].scatter(X_plot,y_pred, c='red')


                X = np.array(dens).reshape(-1,1)
                y = np.array(np.log(speed)).reshape(-1,1)
                reg3 = LM.LinearRegression().fit(X,y)
                score = reg3.score(X,y)
                y_pred = reg3.predict(X)
                y_plot = [math.exp(y_pred[i]) for i in range(len(y_pred))]
                y_plot = np.array(y_plot)

                print('Underwood', score)
                axes[1, 0].scatter(X,y_plot,  c='green')

                fig.show()
                time.sleep(5)
            else:
                fig.show()