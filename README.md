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
 - FCD.py: plots and analyses the trajectories + matches them to the correct lane based on the continuation of their path + couples the trajectory to the stop lights that affected it + estimate resulting shockwaves. 

## Analysis of simulated data
The vehicle records, the network layout and the signal changes are produced by the VISSIM-simulation. There are 3 different regimes, resulting in 60 different periods (20/ regime)

The scripts used are:
 - main_training.py: regulates data delivery process: 
   - Regulates new incoming data: transform .vhp (vehicle records) into .csv files
   - Save file per trajectory in correct folder / period (using simulated_data.py)
   - Runs trajectory_data.py to generate the training data
   - Trains the model (see next section)
 - simulation_data.py: has multiple functions:
   - The vehicle records are saved per vehicle in new csv-files
   - The traffic signals are loaded
   - The network is loaded
   - The vehicle records are analysed: plots of their entire path and parts of their path are made; the vehicle flow and turning fractions are determined per link
 - Trajectory_data.py: analysis of the vehicle records and decomposing them into variables + creating ground truth:
   - The definition of the ground truth based on the full set of trajectories
   - The definition of the variables for every trajectory such as number of stops, travel time on the intersection, shockwave intersection, etc. 
   - Assignment of the trajectories to the respective cycle
   - Saving the results to a csv-file per link and lane
  - data_analysis.py: 
   - Analyse the trajectory variables: density plots, boxplots, correlation, VIF-values, etc. 
 
## HMM
Finally, the HMM is implemented. Two different models are implemented: a first model solely consists of a multinomial logistic regression model, whereas a second model uses an MLR to generate transition probabilities and calculates a time series using a first-order markov chain. The variable 'downstream state' consists of the state estimation on the downstream links, and thus requires an iterative training mechanism: first the model is trained without this variable for all links, and next the model is trained with the variable for only the researched link. 

The scripts used are:
 - main_training.py: 
    - define the saving locations of the models and makes sure the models are run in the correct order
    ! easiest to skip some steps for quick results: early data collection steps (discussed in earlier section) take some time and intermediate results are saved in data. 
 - logistic_regression.py: 
    - Train the MLR without downstream variable based on 45 training periods & save results in map 'Models'
    - Test the MLR without downstream variable for varying penetration rates based on 15 test periods & analyse and plot resulting predictions
 - DownstreamstateLR.py:
    - Run the models predicted in the MLR without downstream variable for the downstream links, resulting in a combined score for the downstream state
 - logistic_regression_withDownstream.py: 
    - Train the MLR with downstream variable based on 45 training periods & save results in map 'Models'
    - Test the MLR with downstream variable for varying penetration rates based on 15 test periods & analyse and plot resulting predictions        
 - logistic_regression1PC.py: only difference with above = random selection of the vehicles
    - Train the MLR without downstream variable based on 45 training periods & save results in map 'Models'
    - Test the MLR without downstream variable for 1 vehicle per cycle based on 15 test periods & analyse and plot resulting predictions
 - DownstreamstateLR1PC.py:
    - Run the models predicted in the MLR without downstream variable for 1 vehicle per cycle for the downstream links, resulting in a combined score for the downstream    state
 - logistic_regression1PC_withDownstream.py: 
    - Train the MLR with downstream variable for one vehicle per cycle based on 45 training periods & save results in map 'Models'
    - Test the MLR with downstream variable for one vehicle per cycle based on 15 test periods & analyse and plot resulting predictions 
 - HMM.py:
    - Train the HMM without downstream variable based on 45 training periods & save results in map 'Models'
      - done by dividing the dataset based on the state in the previous cycle, and fitting a separate MLR to every subset
    - Test the HMM without downstream variable for varying penetration rates based on 15 test periods & analyse and plot resulting predictions
      - Select random number of trajectories, make a time series of observations
      - Use a Markov model, where every cycle new transition probabilities are determined by the observations in that cycle
 - DownstreamstateHMM.py:
    - Run the models predicted in the HMM without downstream variable for the downstream links, resulting in a combined score for the downstream state
 - HMM_withDownstream.py:
    - Train the HMM with downstream variable based on 45 training periods & save results in map 'Models'
       - done by dividing the dataset based on the state in the previous cycle, and fitting a separate MLR to every subset
    - Test the HMM with downstream variable for varying penetration rates based on 15 test periods & analyse and plot resulting predictions
       - Select random number of trajectories, make a time series of observations
       - Use a Markov model, where every cycle new transition probabilities are determined by the observations in that cycle
 - HMM1PC.py:
    - Train the HMM without downstream variable based on 45 training periods & save results in map 'Models'
       - done by dividing the dataset based on the state in the previous cycle, and fitting a separate MLR to every subset
    - Test the HMM without downstream variable for oen vehicle per cycle based on 15 test periods & analyse and plot resulting predictions
       - Select random number of trajectories, make a time series of observations
       - Use a Markov model, where every cycle new transition probabilities are determined by the observations in that cycle
 - DownstreamstateHMM1PC.py:
    - Run the models predicted in the HMM without downstream variable for one vehicle per cycle for the downstream links, resulting in a combined score for the downstream state
 - HMM1PC_withDownstream.py:
    - Train the HMM with downstream variable based on 45 training periods & save results in map 'Models'
       - Done by dividing the dataset based on the state in the previous cycle, and fitting a separate MLR to every subset
    - Test the HMM with downstream variable for one vehicle per cycle based on 15 test periods & analyse and plot resulting predictions
       - Select random number of trajectories, make a time series of observations
       - Use a Markov model, where every cycle new transition probabilities are determined by the observations in that cycle
 - Nullmodel.py: 
    - Determine ratio of occurrences of the states in the training dataset 
    - Determine performance metrics based on the ratios

## Remarks
In general, the current scripts are not very user-friendly. It is therefore advised to read through the files first, as the correct order of running the files is indicated there. Sometimes a piece of code needs to be uncommented in order to run the script separately, since other scripts refer back to those scripts, leading to a long computation time. If it could be solved by saving the intermediate results, it was often done so. These results are saved in the data-map. 
