# Allenurop
This repository contains the scripts and results relating to Sean Condon's work on the Allen Algorithm for LHCb during the summer of 2020.<br/>
<br/>
## Contents Overview
**writeups** - Latex source code and pdf's describing weekly progress on the project.<br/>
<br/>
**catBoostClassifiers** - Directory where trained BDT models are saved along with result plots and files.<br/>
<br/>
**AllenMVA** - Contains training and testing data for BDT models.<br/>
<br/>
**pycharmScripts** - Python scripts for training and testing models.<br/>
<br/>
## Training Custom Models
The files trainGenericClassifier.py and trainSpecificClassifier.py found in pycharmScripts/can be used to train custom BDT models with various hyperparameters from command-line. GenericClassifier.py will train one BDT model for the training data, while SpecificClassifier.py will train a different BDT model for each type of decay in the training data. <br/>
<br/>
Generic classifiers enforce the parent particle PT > 2 Gev, Tau > 0.2 ps, and eta between 2 and 5; this removes around 92% of training data. Specific classifiers do not enforce these cuts because there is not enough training data to train seperate models for each decay type after cutting out so much data. <br/>
<br/>
To make a custom classifier, clone the repository and navigate cd into pycharmScripts. Activate the virtual environment that contains the necessary dependencies with:<br/>
```source venv/bin/activate```<br/>
<br/>
Once the python virtual environment is activated, simply call the python file you want to run, for example:<br/>
```python trainGenericClassifier.py ```<br/>
<br/>
This call comes with a lot of custom arguments that influence the learning rate, iterations, shuffling, and many more hyperparameters of the model training process. You can view the possible arguments with:<br/>
```python trainGenericClassifier.py -h ```<br/>
<br/>
For example, you can train a generic classifier with a learning rate of 0.01, 800 iterations, and a depth of 8 with:<br/>
```python trainGenericClassifier.py --learningRate 0.01 --iterations 800 --depth 8```<br/>


