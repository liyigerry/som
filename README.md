# Predicting cytochrome P450 metabolism by automatically aggregating structural information with graph-based deep learning

## Description

The main work of this paper is as follows: (1) Based on publicly available datasets, three different sets of descriptors based on atomic, 2D topology-based and extracted by recurrent neural networks are designed, and datasets based on these three different sets of descriptors are constructed for use in machine learning and graph neural networks. (2) Four different machine learning models (logistic regression, support vector machine, random deep forest, and decision tree) were constructed and trained on the datasets based on three different sets of descriptors, and the prediction performance of these four models on the three simple sets of descriptors was explored and compared. (3) Four different graph neural network models were constructed and trained on three different descriptor-based datasets built in this paper to explore and compare the prediction performance of the different graph neural networks. The final conclusion shows that graph neural networks using atom-based descriptors can match the prediction results of random forests using 2D topology descriptors, which proves that graph neural networks are capable of extracting structural information. Also, optimal prediction results were achieved by combining descriptors extracted through recurrent neural networks with atom-based descriptors.

This repository stores the corresponding code.

## Prerequisites

This experiment relies on the following packages：

Java 11

Python 3.8

Rdkit. Read file information, calculate descriptors, etc.

CDK 2.5. Computing atomic-based descriptors.

Pytorch 1.11. For building neural network models.

Pytorch Geometric. Used to build graph neural network models.

## Folder Details

NN：This folder holds all the neural network models in this paper.

OriginDataset： This folder stores the original public dataset.

Similiarity， Statistic： These two folders store the code to read and calculate information about the dataset.

descriptor：This folder stores the code to calculate the corresponding descriptors.
