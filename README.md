# Neural Causal Modelling
Graphical Modelling of Neural Networks for determining causality between input, intermediate and output variables

## Background
As machine learning models take over real world tasks, there is an increasing requirement for being able to 
interpret why the model predicts what it does.
This repository implements a framework to get causal interpretations of an Artificial Neural Network for determining neuron and layer level causality.

## Contributions
> Interpretation  of a Neural Network as a Structural Causal Model

> An efficient dictionary-based method to marginalize over the distribution generated on building the structural causal model

> A working pipeline to delineate causal variables in the model from those with spurious correlation

## Workflow 
The workflow for determining causal variables for a particular prediction of the neural network is shown below. For example, you might want to understand why the neural network predicted that a particular image was a show.

<p align="center">
  <img src="./images/workflow.png" width="600"> 
</p>


## Results
### MNIST
To demonstrate how our algorithm can seperate causal variables from spurious and non causal variables, we conduct a study comparing causal variables predicted by our algorithm with the Pearson Correlation Coefficient of each input variable with the output, as shown in Figure 1. We also verify this numerically via an ablation study. 

  <img src="./images/MNIST.jpg" width="500"> 
  <caption>Figure 1 : Causal Variables (Bottom) vs Spurious Variables (Top, in green) for MNIST  </caption>


### HELOC Dataset
The [HELOC Dataset](https://community.fico.com/s/explainable-machine-learning-challenge?tabset-3158a=2) (Home Equity Line of Credit) is an anonymized dataset provided by FICO.
The fundamental task is to predict credit risk. A simple ANN is trained for this, reaching 70% validation accuracy. Causal input variables and their ranges are found using the pipeline above.
 

  <img src="./images/HELOC.jpg" width="500"> 
  <caption>Figure 2: Causal Variables and ranges for HELOC</caption>
<br/><br/>
Figure 1 and 2 both show how variables causal for a particular prediction are delineated by the algorithm. For example, Figure 2 shows that the variable 'External Risk Estimate' being between 42.5 to 52.8 is causal for a bad credit risk prediction.


## Related Literature

1. Pearl, Judea. "An introduction to causal inference." The international journal of biostatistics 6, no. 2 (2010).
2. Chattopadhyay, Aditya, Piyushi Manupriya, Anirban Sarkar, and Vineeth N. Balasubramanian. "Neural Network Attributions: A Causal Perspective." arXiv preprint arXiv:1902.02302 (2019).
3. Harradon, Michael, Jeff Druce, and Brian Ruttenberg. "Causal learning and explanation of deep neural networks via autoencoded activations." arXiv preprint arXiv:1802.00541 (2018).
4. Molnar, Christoph. Interpretable machine learning. Lulu. com, 2019.


