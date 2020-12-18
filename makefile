CC = g++
STD = -std=c++11
LLDB = -g
BOOST = -lboost_serialization
PYTHON = -lboost_python-py36
FMATH = -fomit-frame-pointer -fno-operator-names -msse2 -mfpmath=sse -march=native
GFLAGS = -lglog -lgflags
#-I/root/boost -L/root/boost/stage/lib -I/usr/include/boost/ 
INCLUDE = -I/usr/include/ -pthread
LDFLAGS = `python3-config --includes` `python3-config --ldflags`

trainer:
	$(CC) -O3 $(STD) -o bin/cstm cstm/model.cpp $(BOOST) $(INCLUDE) $(FMATH) $(GFLAGS) 

install:
	$(CC) -O3 $(STD) -DPIC -shared -fPIC -o pycstm.so pycstm.cpp $(INCLUDE) $(LDFLAGS) $(PYTHON) $(BOOST) $(FMATH) $(GFLAGS)

prepare:
	$(CC) word2vec/word2vec.cpp -o bin/word2vec -lm -pthread -O3 -march=native -Wall -Wextra -funroll-loops -Wno-unused-result

distance:
	$(CC) word2vec/distance.cpp -o bin/distance -lm -pthread -O3 -march=native -Wall -Wextra -funroll-loops -Wno-unused-result

test:
	$(CC) -O3 -Wall -o a.out cstm/model.cpp $(LLDB) $(BOOST) $(FMATH) $(GFLAGS)

clean:
	rm -f bin/cstm

.PHONY: clean