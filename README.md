# CNN-EEG
Applying Convolutional Neural Networks to EEG signal Analysis




The aim of this project is to build a Convolutional Neural Network (CNN) model for processing and classification of multi-electrode EEG signal. This model was designed for incorporating EEG data collected from 7 pair of symmetrical electrodes. The MindBigData EPOH dataset (can be downloaded from **[here](http://mindbigdata.com/opendb/MindBigData-EP-v1.0.zip)**, 2,66 GB) was used to train the models. This projects presents architectures for a multiclass (10) and binary (one-versus-all) classifiers. We used Pytorch API for bulding the CNN network and the Scikit-learn library to supplement our data processing and performance analysis methods. 


* `dataLoader.py`: The dataloader that creates a Pytorch compatible tensor from the raw tab-separated txt file. Dataloader has to be ran to produce a tensor that would then be used as an input for one of 4 CNN models (convNet, lowKernelNet, thinNet, or thinNet).
* `convNet.py`: the main 6-layer CNN architechture
* `lowKernelNet.py` An alternative reduced architecture
* `thinNet.py`:  An alternative reduced architecture
* `deployCNN.py`: The pipeline for training and testing the muplticlass classifier models. The script is designed in a way that makes it easier to compare the perfomance metrics for models built on different versions of the dataset (in our case, datasets with different pairs of symmetrical channels excluded). Detailed instructions on how to run the model are provided in the script file.
