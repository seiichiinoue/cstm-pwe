#include <boost/python.hpp>
#include <string>
#include <iostream>
#include <vector>
using namespace std;
using namespace boost;

void string_to_wstring(const std::string &src, std::wstring &dest) {
	wchar_t *wcs = new wchar_t[src.length() + 1];
	mbstowcs(wcs, src.c_str(), src.length() + 1);
	dest = wcs;
	delete [] wcs;
}

int load_vector(string fname, vector<wstring> &vocab_list, vector<vector<double>> &word_vec) {
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
    int resw; resw = fscanf(f, "%lld", &words);
    int ress; ress = fscanf(f, "%lld", &size);

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
    int tmp;
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
            tmp = fread(&M[a + b * size], sizeof(float), 1, f);
            if (a < sizes)
                M1[a + b * sizes] = M[a + b * size];
            else
                M2[(a - sizes) + b * sized] = M[a + b * size];
        }
    }
    fclose(f);
    
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
    word_vec.resize(words, vector<double>(sized, 0));
    for (b=0; b<words; ++b) {
        // words
        string s = &vocab[b * max_w];        
        wstring ws;
        string_to_wstring(s, ws);
        vocab_list[b] = ws;
        // word vector
        for (a=0; a<size; ++a) {
            if (a < sizes) {
                double tmp1 = M1[a + b * sizes];
                word_vec[b][a] = tmp1;
            }
        }
    }
    return 0;
}

python::dict get_word_vectors(string filename) {
    vector<wstring> vocab;
    vector<vector<double>> semantic_vec;
    load_vector(filename, vocab, semantic_vec);
    python::dict dic;
    for (int i=0; i<vocab.size(); ++i) {
        python::list vector_list;
        vector<double> vector = semantic_vec[i];
        for (int i=0; i<semantic_vec[i].size(); ++i) {
            vector_list.append(vector[i]);
        }
        dic[vocab[i]] = vector_list;
    }
    return dic;
}

// python::list get_word_vectors(string filename) {
//     vector<wstring> vocab;
//     vector<vector<double>> semantic_vec;
//     load_vector(filename, vocab, semantic_vec);
//     python::list list;
//     for (int i=0; i<vocab.size(); ++i) {
//         python::list vector_list;
//         vector<double> vector = semantic_vec[i];
//         for (int i=0; i<semantic_vec[i].size(); ++i) {
//             vector_list.append(vector[i]);
//         }
//         list.append(vocab[i]);
//         list.append(vector_list);
//     }
//     return list;
// }

BOOST_PYTHON_MODULE(pyvec) {
    def("get_word_vectors", get_word_vectors);
}