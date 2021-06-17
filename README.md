# Thesis: Queue Spillover Detection at Urban Signalised Intersections using Floating Car Data
This repository consists of the python scripts written during my master's thesis at KU Leuven for the Master in Logistics and Traffic. It discusses the detection of queue spillovers at signalised intersections based on floating car data (FCD). Spillovers occur when a queue from a downstream link spills back onto the upstream link, resulting in a blockage of the upstream intersection. It aims to analyse the traffic state using a Hidden Markov Model (HMM). This HMM receives information from trajectory observations and circumstantial information. Based on these observations, a multinomial logistic regression model (MLR) predicts the transition probabilities from the state in the previous time step to the state in the current time step. The states that are distinguished are: undersaturation, three levels of oversaturation, the spillover state. 

There are two main groups of scripts: 
 1. Scripts used to visualise, analyse and extract variables from the real-life VLOG- and FCD-data.
 2. Scripts used to analyse the vehicle records from the simulated setting.
 3. Scripts used to train and predict the model.

## Real-life data analysis
### a) Intersection Topology Format:
The ITF-files present the intersection topology as xml-files. They are read and processed in a more easily readable way for the further processes. Moreover, a visually matching of the link IDs with the IDs given by Be-Mobile in the FCD is also included in this file. 

Scripts:
 - intersectionLayout.py: read the Intersection Topology Files, such that the layout of the lanes and segments can be used 

Data:
 - ITF-files: .xml-documents for every intersection
 - shapefiles of the intersection
### b) VLOG-data:
The VLOG-data contains data on the traffic signal cycles and the loop detector measurements. They are accompanied by a configuration file and a file describing the layout of the intersection. 

The scripts used to analyse are:
 - VLOG.py: get signal cycles & loop detections + transform detections into fundamental parameters speed, flow and density 
 - FD_regression.py: use the processed data from the LDs to find the best fitting fundamental diagram for every detector

### c) FCD-data:
The FCD-data consists of vehicle trajectories on the network within a certain perimeter at 1 Hz. They are mapmatched to a basemap by Be-Mobile. 

The scripts used are:
 - FCDdata.py: plots and analyses the trajectories + matches them to the correct lane based on the continuation of their path + couples the trajectory to the stop lights that affected it + estimate resulting shockwaves. 

## Analyse simulated data
