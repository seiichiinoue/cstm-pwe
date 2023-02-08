# Modeling Text Using the Continuous Space Topic Model with Pre-Trained Word Embeddings

## Environment

activate docker.

```
$ docker build -t myenv .
$ docker run -it -v [local path]:/workspace/ myenv
$ docker exec -it [container id] /bin/bash
```

## Preparation

```bash
$ mkdir bin/
$ mkdir data/ # data must be located at here.
```

pre-train the word embeddings using CBOW model.

```bash
$ make prepare
$ cd bin/ && ./word2vec -size 200 -size-s 100 -train ../data/data.txt -output ./vec.bin

```

## Training

compile the CSTM and execute training.

```bash
$ make
$ cd bin/ && ./cstm -ndim_d=100 -ignore_word_count=4 -epoch=100 -num_threads=10 -data_path=../data/train/ -validation_data_path=../data/validation/ -vec_path=./vec.bin -model_path=./cstm.model > ../log/cstm.log
```

## Reference

Seiichi Inoue, Taichi Aida, Mamoru Komachi, and Manabu Asai. 2021. [Modeling Text using the Continuous Space Topic Model with Pre-Trained Word Embeddings](https://aclanthology.org/2021.acl-srw.15/). In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: Student Research Workshop, pages 138–147, Online. Association for Computational Linguistics.