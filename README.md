# Deep-Emotion: Facial Expression Recognition Using Attentional Convolutional Network (Tensorflow implementation)
#### \*WIP\*

As a personal exercise on reading and implementing SOTA papers, I implemented of one of the leading state-of-the-art papers in Facial Expression Recoginition (FER), [Deep-Emotion](https://arxiv.org/abs/1902.01019). As far as I know, there is no Tensorflow implementation of the paper so decided to go with TF as my choice of framework.

There are, however, a couple Pytorch versions. The most popular of them is [omarSayed7's non-official implementation of DeepEmotion2019](https://github.com/omarsayed7/Deep-Emotion.git). I forked and used it as a reference.

## Architecture

In a nutshell, the paper proposes an attentional CNN that predicts facial expressions by focussing a classifier layer on the most relevant portions of the input image. This attention mechanism is achieved using a [Spatial Transformer Network](https://arxiv.org/pdf/1506.02025.pdf) or STN in short. The STN works by learning a set of 6 transformation prameters that is then used to perform an affine transformation of the input image. In this implementation, I used [kevinzakka's library](https://github.com/kevinzakka/spatial-transformer-network/tree/master) to perform the spatial transformation.

In the proposed model, a feature extractor works in parallel with the STN to generate a feature map that is fed to a classification layer for emotion inference.

<p align="center">
  <img src="imgs/net_arch.PNG" width="960" title="Deep-Emotion Architecture">
</p>

## Contributions
The model architecture in the Pytorch implementation differs slightly with that described in the paper \(in aspects like input image flow, kernel initialization, regularization, hyperparameters etc\). I tried to mirror the paper as closely as possible and made suitable changes. In additon, I worked with a couple assumptions as I was unsure of certain specifics of the model architecture as described in the paper. For this reason, the implementation might not be exactly what the authors intended, however, I have added comments in the code at all such places explaining my reasons. 

## Datasets
This implementation uses the following datasets:
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [CK+](https://ieeexplore.ieee.org/document/5543262)
- [JAFFE](https://www.researchgate.net/publication/220013358_The_japanese_female_facial_expression_jaffe_database)
- [FERG](https://homes.cs.washington.edu/~deepalia/papers/deepExpr_accv2016.pdf)

## Prerequisites
Make sure you have the following libraries installed:
- tensorflow >= 2.13.0
- stn == 1.0.1
- pandas
- pillow
- tqdm

## Repository Structure
This repository is organized as follows:
- [`main`](/main.py): Contains setup for the dataset and training loop.
- [`deep_emotion`](/deep_emotion.py): Defines the model class.
- [`generate_data`](/generate_data.py): Sets up the [dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Usage
Clone the repository and follow these steps.

### Environment
This repository was tested using ```python==3.9.12``` and ```pip==24.0``` on a Windows machine.
To setup the environment, create and activate a virtual environment (```virtualenv --python=python3.9.12 venv | venv/Scripts/activate```) and run:
```bash
pip install -r requirements.txt
```

### Download the Data
1. Download the dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
2. Decompress `train.csv` and `test.csv` into the `./data` folder within the repo.

### Setup the Dataset
Open terminal and run:
```
python main.py [-s [True]] [-d [data_path]]

--setup                 Setup the dataset for the first time
--data                  Data folder that contains data files
```
For example,
```
python main.py -s True -d data
```
This will produce images out of the .csv files downloaded from Kaggle and split them into training and validation datasets.

### Train the model
Set hyperparameters
```
python main.py [-t] [--data [data_path]] [--hparams [hyperparams]]
              [--epochs] [--learning_rate] [--batch_size]

--data                  Data folder that contains training and validation files
--train                 True when training
--hparams               True when changing the hyperparameters
--epochs                Number of epochs
--learning_rate         Learning rate value
--batch_size            Training/validation batch size
```
For example, to specify your own hyperparameters, run:
```
python main.py -t True -d data -hparams True --epochs 5 --learning_rate 0.005 --batch_size 32
```
To use default hyperparameters (as specified in the paper), run:
```
python main.py -t True -d data
```
## Samples
<p align="center">
  <img src="imgs/samples.png" width="720" title="Deep-Emotion Architecture">
</p>
```
