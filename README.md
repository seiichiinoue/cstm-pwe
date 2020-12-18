# Continuous Space Topic Model with Word Embeddings

## Description

the implementation of Continuous Space Topic Model with Word Embeddings, which is the enhenced model of [CSTM](http://chasen.org/~daiti-m/paper/nl213cstm.pdf) by Daichi Mochihashi.

## Environment

- C++ 14+
- clang++ 9.0
- boost 1.71.0
- glog 0.4.0
- gflag 2.2.2
- boost-python3
- python3

## Usage

- prepare document-based corpus and split it into train dataset and validation dataset

- training ETM with MCMC.

```bash
$ make
$ ./cstm -ndim_d=20 -ignore_word_count=4 -epoch=100 -num_threads=1 -data_path=./data/train/ -validation_data_path=./data/validation/ -model_path=./model/cstm.model
```

## Reference

- [Modeling Text through Gaussian Processes](http://chasen.org/~daiti-m/paper/nl213cstm.pdf)
