# HML4Rec
The source code of paper "HML4Rec: Hierarchical Meta-Learning for Cold-Start Recommendation in Flash-Sale E-commerce" for reproduction.

## Reproduce Evaulation Results in the Paper
Experiment.ipynb gives the code for reproducing our experiment on MovieLens 1M. 


### Requirements
- python 3.7+
- pytorch 1.1+
- pandas 1.1+
- numpy 1.19+
- scipy 1.5+

### Preparing data
```python
train_order, valid_order, test_order, movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item = prepare_dataset()
```
- ```train_order, valid_order, test_order```: Training set, validation set, and test set.
- ```movie_dict```: the dictionary of movie id -> movie side features.
- ```user_dict```: the dictionary of user id -> user side features.
- ```Tr_rated_dict```: the dictionary of interaction records in training set, user id -> movie list.
- ```Rated_dict```: the dictionary of interaction records in whole dataset, user id -> movie list.
- ```condidate_item```: the condidate recommendation movie set.

### Initializing a model
```python
RecSys = HML4Rec(configs, train_order, valid_order, test_order, movie_dict, user_dict, Tr_rated_dict, Rated_dict, condidate_item)
```

### Training
```python
RecSys.train()
```

### Generating results and evaluating
The recommendation results are in the file: ```Rec_result```, and the evaluation results are in the file: ```Figures```.

```python
RecSys.Recommending()
Show_result()
```

## Acknowledge

Some portions of this repo is borrowed from the following repos:
- [MeLU](https://github.com/hoyeoplee/MeLU)
- [MAMO](https://github.com/dongmanqing/Code-for-MAMO)
