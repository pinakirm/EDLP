Entropy-Guided Sampling of Flat Modes in Discrete Spaces

This repository contains code for the paper
Entropy-Guided Sampling of Flat Modes in Discrete Spaces.



# Introduction
We propose Entropic DLP (EDLP), an entropy-guided,
gradient-based proposal for sampling discrete flat
modes. EDLP efficiently incorporates local entropy
guidance by coupling discrete and continuous variables
within a joint distribution.


# Dependencies
* [PyTorch 1.9.1](http://pytorch.org/) 
* [torchvision 0.10.1](https://github.com/pytorch/vision/)

# Usage

## Sampling From 4D Joint Bernoulli
Please run
```
cd Bernoulli
python bernoulli_sample.py --sampler=<SAMPLER>
```

## Sampling From Ising Models
Please run
```
python ising_sample.py --sampler=<SAMPLER>
```
## Sampling From Restricted Boltzmann Machines
Please run
```
python rbm_sample.py --sampler=<SAMPLER>
```
## Travelling Salesman Problem
Enter Directory
```
./TSP
```
Then run
```
python TSP.py --sampler=<SAMPLER> 
```

## Binary Bayesian Neural Networks
See 
```
cd BinaryBNN
python bayesian.py --sampler=<SAMPLER>
```



# References
* This repo is built upon the [DLP repo](https://github.com/ruqizhang/discrete-langevin) 
