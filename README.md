# Modeling Text Using the Continuous Space Topic Model with Pre-Trained Word Embeddings

## Usage

- activate docker 

```
$ docker build -t myenv .
$ docker run -it -v [local path]:/workspace/ myenv
$ docker exec -it [container id] /bin/bash
```

- compile

```bash
$ make prepare
$ make
$ mkdir bin/
$ mkdir data/ # data must be located at here.
```


- pre-train the word embeddings using CBOW model.

```bash
$ cd bin/ && ./word2vec -size 200 -size-s 100 -train ../data/data.txt -output ./vec.bin

```

- compile the CSTM and execute training.

```bash
$ cd bin/ && ./cstm -ndim_d=100 -ignore_word_count=4 -epoch=100 -num_threads=10 -data_path=../data/train/ -validation_data_path=../data/validation/ -vec_path=./vec.bin -model_path=./cstm.model > ../log/cstm.log
```

