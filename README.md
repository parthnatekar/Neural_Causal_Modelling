# Neural_Causal_Modelling
Graphical Modelling of Neural Networks for determining causality between input, intermediate and output variables

## Background
As machine learning models take over real world tasks, there is an increasing requirement for being able to 
interpret why the model predicts what it does.
This repository implements a framework to get causal interpretations of an Artificial Neuron Network for determining neuron and layer level causalit.

## Workflow 
<p align="center">
  <img src="./images/workflow.png"> 
</p>


## Results

### HELOC Dataset
The [HELOC Dataset](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2) (Home Equity Line of Credit) is an anonymized dataset provided by FICO.
The fundamental task is to predict credit risk. A simple ANN is trained for this, reaching 70% validation accuracy. Causal input variables and their ranges are found using the pipeline above.
