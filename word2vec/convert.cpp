#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// # include <malloc/malloc.h>      for macos
#include <malloc.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <iostream>
#include <ostream>
using namespace std;

const long long max_size = 2000; // max length of strings
const long long N = 15;          // number of closest words that will be shown
const long long max_w = 50;      // max length of vocabulary entries


void string_to_wstring(const std::string &src, std::wstring &dest) {
	wchar_t *wcs = new wchar_t[src.length() + 1];
	mbstowcs(wcs, src.c_str(), src.length() + 1);
	dest = wcs;
	delete [] wcs;
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int load_vector(string fname, vector<wstring> &vocab_list, vector<vector<double>> &semantic_vec, vector<vector<double>> &stylistic_vec) {
    long long max_size = 2000, N = 15, max_w = 50;
    FILE *f;
    char st1[max_size];
    char *bestw[N];
    char st[100][max_size];
    float dist, len, len1, len2, bestd[N], vec[max_size];
    long long words, size, a, b, c, d, cn, bi[100];
    long long sizes = -1, sized;
    float *M, *M1, *M2;
    char *vocab, *file_name[max_size];
    f = fopen(fname.c_str(), "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);

    /* size of stylistic (sizes) and syntactic/semantic (sized) vector */
    if (sizes == -1) {
        sizes = size / 2;
    }
    sized = size - sizes;
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    for (a = 0; a < N; a++)
        bestw[a] = (char *)malloc(max_size * sizeof(char));
    M = (float *)malloc((long long)words * (long long)size * sizeof(float));
    M1 = (float *)malloc((long long)words * (long long)sizes * sizeof(float)); //stylistic
    M2 = (float *)malloc((long long)words * (long long)sized * sizeof(float)); //syntactic/semantic
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }
    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' '))
                break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n'))
                a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) {
            fread(&M[a + b * size], sizeof(float), 1, f);
            if (a < sizes)
                M1[a + b * sizes] = M[a + b * size];
            else
                M2[(a - sizes) + b * sized] = M[a + b * size];
        }
    }
    fclose(f);
    // printf("vocab size: %lld\n", words);
    // for (b=0; b<words; ++b) {
    //     printf("%s\n", &vocab[b * max_w]);
    // }
    
    /* language code settings */
    setlocale(LC_CTYPE, "ja_JP.UTF-8");
    ios_base::sync_with_stdio(false);
    locale default_loc("ja_JP.UTF-8");
    locale::global(default_loc);
    locale ctype_default(locale::classic(), default_loc, locale::ctype);
    wcout.imbue(ctype_default);
    wcin.imbue(ctype_default);

    /* convert array to vector */
    vocab_list.resize(words, L"");
    semantic_vec.resize(words, vector<double>(sized, 0));
    stylistic_vec.resize(words, vector<double>(sizes, 0));
    for (b=0; b<words; ++b) {
        // words
        string s = &vocab[b * max_w];
        wstring ws;
        string_to_wstring(s, ws);
        vocab_list[b] = ws;
        // word vector
        for (a=0; a<size; ++a) {
            // double tmp = M[a+b+size];
            // word_vec[b][a] = tmp;
            if (a < sizes) {
                double tmp1 = M1[a + b * sizes];
                stylistic_vec[b][a] = tmp1;
            } else {
                double tmp2 = M2[(a - sizes) + b * sized];
                semantic_vec[b][a-sizes] = tmp2;
            }
        }
    }
    // wcout << vocab_list[101] << endl;
    cout << "word_vector size: " << size << " stylistic_vector size: " << sizes << " semantic_vector size: " << sized << endl;
    return 0;
}

int main() {
    vector<wstring> vocab;
    vector<vector<double>> semantic_vec, stylistic_vec;
    load_vector("../bin/vec_dim300.bin", vocab, semantic_vec, stylistic_vec);
    // look size of vector
    double scaleing_coef = 0;
    double max_scale = 0, min_scale = 1e9; 
    double mean = 0;
    for (int ind=0; ind<semantic_vec.size(); ++ind) {
        double size = 0;
        double inner = 0;
        vector<double> &tar = semantic_vec[ind];
        for (int i=0; i<tar.size(); ++i) {  // tar_size = 300
            // cout << tar[i] << endl;
            // size += tar[i] * tar[i];
            if (max_scale < tar[i]) max_scale = tar[i];
            if (min_scale > tar[i]) min_scale = tar[i];
            size += tar[i];
            inner += tar[i] * tar[i];
        }
        scaleing_coef += inner;
        // cout << "size: " << size << endl;
        mean += (double)(size/(double)tar.size());
    }
    scaleing_coef = sqrt(scaleing_coef / (double)semantic_vec.size());
    
    cout << "all mean: " << mean / (double)semantic_vec.size() << endl;
    cout << "all max: " << max_scale << " all min: " << min_scale << endl;
    cout << "scaling coef: " << scaleing_coef << endl;
    
    // scale vector
    for (int ind=0; ind<semantic_vec.size(); ++ind) {
        vector<double> &tar = semantic_vec[ind];
        for (int i=0; i<tar.size(); ++i) {  // tar_size = 300
            tar[i] /= scaleing_coef;
            tar[i] /= 300.0;  // dim size
        }
    }
    double norm = 0;
    double max_norm = 0;
    for (int ind=0; ind<semantic_vec.size(); ++ind) {
        vector<double> &tar = semantic_vec[ind];
        double tmp = 0;
        for (int i=0; i<tar.size(); ++i) {  // tar_size = 300
            tmp += sqrt(tar[i] * tar[i]);
        }
        norm += tmp;
        if (tmp > max_norm) max_norm = tmp;
    }
    norm = norm / (double)semantic_vec.size();
    cout << "scaled norm: " << norm << endl;    // var;
    cout << "max scaled norm: " << max_norm << endl;

}
