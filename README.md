# Thesis-Spillover-Detection
This repository consists of the python scripts written during my master's thesis at KU Leuven for the Master in Logistics and Traffic. It discusses the detection of queue spillovers at signalised intersections based on floating car data (FCD). Spillovers occur when a queue from a downstream link spills back onto the upstream link, resulting in a blockage of the upstream intersection. It aims to analyse the traffic state using a Hidden Markov Model (HMM). This HMM receives information from trajectory observations and circumstantial information. Based on these observations, a multinomial logistic regression model (MLR) predicts the transition probabilities from the state in the previous time step to the state in the current time step. The states that are distinguished are: undersaturation, three levels of oversaturation, the spillover state. 

There are two main groups of scripts: 
 1. Scripts used to visualise, analyse and extract variables from the real-life VLOG- and FCD-data.
 2. Scripts used to analyse the vehicle records from the simulated setting.
 3. Scripts used to train and predict the model.

## Real-life data analysis
