# SiMVC & CoMVC

The project extends on the base of the implementations of SiMVC and CoMVC, presented in the paper:

"Reconsidering Representation Alignment for Multi-view Clustering" by
Daniel J. Trosten, Sigurd Løkse, Robert Jenssen and Michael Kampffmeyer, in _CVPR 2021_.

BibTeX:
```text
@inproceedings{trostenMVC,
  title        = {Reconsidering Representation Alignment for Multi-view Clustering},
  author       = {Daniel J. Trosten and Sigurd Løkse and Robert Jenssen and Michael Kampffmeyer},
  year         = 2021,
  booktitle    = {2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```


## Installation
Python version 3.8

To install the required packages, run:
```
pip install -r requirements.txt
```


## Datasets
The following four datasets have been included, preprocessed, experimented in this extended project:

- `chestx`
- `rgbd` (RGB-D)  
- `coil`
- `fmnist`

## Experiment configuration
Experiment configs are nested configuration objects, where the top-level config is an instance of 
`config.defaults.Experiment`. 

The configuration object for the contrastive model on E-MNIST, for instance, looks like this:
```Python
from config.defaults import Experiment, CNN, DDC, Fusion, Loss, Dataset, CoMVC, Optimizer


mnist_contrast = Experiment(
    dataset_config=Dataset(name="mnist_mv"),
    model_config=CoMVC(
        backbone_configs=(
            CNN(input_size=(1, 28, 28)),
            CNN(input_size=(1, 28, 28)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=10),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|contrast",
            # Additional loss parameters go here
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-3,
            # Additional optimizer parameters go here
        ) 
    ),
    n_epochs=100,
    n_runs=20,
)
```

However, we have added one line to the `use_attention=True` to control if we want to use the extended mechanism. The default configuration is to use attention.

The following training and experment coomands are the same with the based projects.

## Running an experiment
In the `src` directory, run:
```
python -m models.train -c <config_name> 
```
where `<config_name>` is the name of an experiment config from one of the files in `src/config/experiments/` or from 
'src/config/eamc/experiments.py' (for EAMC experiments).

### Overriding config parameters at the command-line
Parameters set in the config object can be overridden at the command line. For instance, if we want to change the 
learning rate for the E-MNIST experiment below from 0.001 to 0.0001, and the number of epochs from 100 to 200,
we can run:
```
python -m models.train -c mnist_contrast \
                       --n_epochs 200 \
                       --model_config__optimizer_config__learning_rate 0.0001
```
Note the double underscores to traverse the hierarchy of the config-object.

## Evaluating an experiment
Run the evaluation script:
```Bash
python -m models.evaluate -c <config_name> \ # Name of the experiment config
                          -t <tag> \         # The unique 8-character ID assigned to the experiment when calling models.train
                          --plot             # Optional flag to plot the representations before and after fusion.
```

