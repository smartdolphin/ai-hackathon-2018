# Naver AI Hackathon 2018
[Naver AI Hackathon 2018](https://github.com/naver/ai-hackathon-2018) task was to predict movie ratings and the similarities between two questions.<br/>
I participated as a team `Deeppangyo`.

## Summary of approach
- Movie rating prediction: **Ranked 9th (over 200 teams, MSE: 2.86229)**
- Two question similarity prediction: **Ranked 8th (1st round) / 11th (2nd round) (ACC: 0.960798/0.960328)**

## Features
#### Movie rating prediction
- Fully dockerized environments
- Bidirectional LSTM (Phoneme + Char + Word)
- L2 regularization
- Word CNN
- Self Attention (but not used)

#### Two question similarity prediction
- Siamese network using Manhattan distance
- Contrastive loss
- Convolution + Bidirectional GRU
- Self Attention
- GAP/GMP
- Word CNN (but not used)

## Usage
#### Movie rating prediction on NSML

Login with `nsml`, and run commands as follows:

````bash
$ cd movie-review
$ nsml run -d movie_final -e main.py -a "--epochs 10 --batch 1000 --strmaxlen 117 --embedding 64 --dropout 0.3"
````

#### Two question similarity prediction on NSML

Login with `nsml`, and run commands as follows:
````bash
$ cd kin
$ nsml run -d kin_final -e main.py -a "--epochs 200 --batch 2000 --strmaxlen 250 --embedding 128 --valrate 0.3"
````

## Requirements
- python 3
- keras
- torch
- numpy
- matplotlib
- tensorflow
- pandas
- sklearn