# CNN-EEG
Applying Convolutional Neural Networks to EEG signal Analysis



The aim of this project is to build a Convolutional Neural Network (CNN) model for processing and classification of multi-electrode EEG signal. This model was designed for incorporating EEG data collected from 7 pair of symmetrical electrodes. The MindBigData EPOH dataset (can be downloaded from here, 2,66 GB) was used to train the models. This projects presents architectures for a multiclass (10) and binary (one-versus-all) classifiers. We used Pytorch API for bulding the CNN network and the Scikit-learn library to supplement our data processing and performance analysis methods.

dataLoader.py: the dataloader that creates a Pytorch compatible tensor from the raw tab-separated txt file. Dataloader has to be ran to produce a tensor that would then be used as an input for one of 4 CNN models (convNet, lowKernelNet, thinNet, or thinNet).
convNet.py: the main 6-layer CNN architechture
lowKernelNet.py An alternative reduced architecture
thinNet.py: An alternative reduced architecture
deployCNN.py: The file to run all of our operations for multiclass classification. Detailed instructions on how to run the model are provided in the script file.
binaryV2.py a binary CNN classifier. To run the model, uncomment the lines at the bottom of the code file to train and test the convNet binary classifier for a digit of interest.
rfModel.py: an (unsuccessful) attempt to build a random forest classifier for analyzing EEG data
tensorToNumpy.py: creates a numpy data object from a PyTorch tensor; this code was utilized when we attempted to build a random forest classifier.
