# Analysis of Supervised Learning Techniques

Instructions for running the experiments.

Support Vector Machines and KNN are implemented in python using the pybrain and sklearn packages. Because of the different formatting of the two input datasets, I have written 2 python files.

To run these three experiments on the splicing data, using learningAlgorithms.py and to run them on the letter classification data, run learningAlgorithms2.py.

KNN: Number of neighbors cannot be specified as it is running for multiple values. To set a limit for the value, set the parameter ‘neighbors’ to whatever you see fit.

SVM: To test different kernel methods, specify the kernel in the initialization of the SVM. Options are ‘linear, sigmoid and rbf’.

Neural Net: Run this through weka using multilayer perceptron, changing the parameters epochs, learning rate and momentum, leaving the code as is.

Decision tree: Use J48 in weka, varying confidence and whether or not to activate pruning.

Boosting: Run this through weka, using AdaboostM1 with J48 as the tree and varying confidence and pruning.


