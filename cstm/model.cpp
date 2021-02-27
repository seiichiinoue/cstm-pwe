#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <thread>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map> 
#include "cstm.hpp"
#include "vocab.hpp"
using namespace boost;
using namespace cstm;

template<typename T>
struct multiset_comparator {
    bool operator()(const pair<id, T> &a, const pair<id, T> &b) {
        return a.second > b.second;
    }
};

void split_string_by(const string &str, char delim, vector<string> &elems) {
    elems.clear();
    string item;
    for (char ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}
void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems) {
    elems.clear();
    wstring item;
    for (wchar_t ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}

bool ends_with(const std::string& str, const std::string& suffix) {
    size_t len1 = str.size();
    size_t len2 = suffix.size();
    return len1 >= len2 && str.compare(len1 - len2, len2, suffix) == 0;
}

class CSTMTrainer {
public:
    CSTM *_cstm;
    Vocab *_vocab;    
    vector<vector<vector<id>>> _dataset;
    vector<vector<vector<id>>> _validation_dataset;
    vector<unordered_set<id>> _word_ids_in_doc;
    vector<unordered_set<id>> _word_ids_in_doc_validation;
    vector<int> _sum_word_frequency;                                // word frequency per document
    vector<int> _sum_word_frequency_validation;
    vector<id> _random_word_ids;
    vector<int> _random_doc_ids_for_semantic;
    unordered_map<id, unordered_set<int>> _docs_containing_word;    // word -> [doc_ids]
    unordered_map<id, int> _word_frequency;
    unordered_map<string, int> _doc_filename_to_id;
    unordered_map<int, string> _doc_id_to_filename;
    
    double *_old_vec_copy;
    double *_new_vec_copy;
    double **_old_vec_copy_thread;
    double **_new_vec_copy_thread;
    double *_old_alpha_words;
    double *_Zi_cache;
    
    int _num_threads;
    int _ndim_d;
    int _ignored_vocabulary_size;
    int _num_of_ignored_word;
    std::thread* _semantic_doc_threads;

    // stat
    // times of acceptance in MH estimation
    int _num_acceptance_doc_in_semantic_space;
    int _num_acceptance_word;
    int _num_acceptance_alpha0;
    // times of rejection 
    int _num_rejection_doc_in_semantic_space;
    int _num_rejection_word;
    int _num_rejection_alpha0;
    // times of sampling
    int _num_word_vec_sampled;
    int _num_doc_vec_in_semantic_space_sampled;
    
    int _random_sampling_word_index;
    int _random_sampling_doc_vec_in_semantic_space_index;
    unordered_map<id, int> _num_updates_word;
    unordered_map<int, int> _num_updates_doc_vec_in_semantic_space;

    CSTMTrainer() {
        // lang code settings
        setlocale(LC_CTYPE, "ja_JP.UTF-8");
        ios_base::sync_with_stdio(false);
        locale default_loc("ja_JP.UTF-8");
        locale::global(default_loc);
        locale ctype_default(locale::classic(), default_loc, locale::ctype);
        wcout.imbue(ctype_default);
        wcin.imbue(ctype_default);
        _cstm = new CSTM();
        _vocab = new Vocab();
        _old_vec_copy = NULL;
        _new_vec_copy = NULL;
        _old_vec_copy_thread = NULL;
        _new_vec_copy_thread = NULL;
        _old_alpha_words = NULL;
        _Zi_cache = NULL;
        _semantic_doc_threads = NULL;
        _ndim_d = 0;
        _num_threads = 1;
        _ignored_vocabulary_size = 0;
        _num_of_ignored_word = 0;
        reset_statistics();
        _random_sampling_word_index = 0;
        _random_sampling_doc_vec_in_semantic_space_index = 0;
    }
    ~CSTMTrainer() {
        delete _cstm;
        delete _vocab;
    }

    void reset_statistics() {
        _num_acceptance_word = 0;
        _num_acceptance_doc_in_semantic_space = 0;
        _num_acceptance_alpha0 = 0;
        _num_rejection_word = 0;
        _num_rejection_doc_in_semantic_space = 0;
        _num_rejection_alpha0 = 0;
        _num_word_vec_sampled = 0;
        _num_doc_vec_in_semantic_space_sampled = 0;
    }

    void prepare(bool validation=false) {
        int num_docs = _dataset.size();
        int vocabulary_size = _word_frequency.size();
        // cache
        _old_vec_copy = new double[_ndim_d];
        _new_vec_copy = new double[_ndim_d];
        _old_vec_copy_thread = new double*[_num_threads];
        _new_vec_copy_thread = new double*[_num_threads];
        for (int i=0; i<_num_threads; ++i) {
            _old_vec_copy_thread[i] = new double[_ndim_d];
        }
        for (int i=0; i<_num_threads; ++i) {
            _new_vec_copy_thread[i] = new double[_ndim_d];
        }
        _semantic_doc_threads = new std::thread[_num_threads];
        // CSTM
        _cstm->initialize(_ndim_d, vocabulary_size, num_docs);
        for (int doc_id=0; doc_id<num_docs; ++doc_id) {
            vector<vector<id>> &dataset = _dataset[doc_id];
            for (int data_index=0; data_index<dataset.size(); ++data_index) {
                vector<id> &word_ids = dataset[data_index];
                for (const id word_id : word_ids) {
                    _cstm->add_word(word_id, doc_id);
                }
            }
            _num_updates_doc_vec_in_semantic_space[doc_id] = 0;
            _random_doc_ids_for_semantic.push_back(doc_id);
        }
        if (validation) {
            for (int doc_id=0; doc_id<num_docs; ++doc_id) {
                vector<vector<id>> &dataset = _validation_dataset[doc_id];
                for (int data_index=0; data_index<dataset.size(); ++data_index) {
                    vector<id> &word_ids = dataset[data_index];
                    for (const id word_id : word_ids) {
                        _cstm->add_word_validation(word_id, doc_id);
                    }
                }
            }
        }
        _cstm->prepare(validation=validation);
        assert(_ndim_d == _cstm->_ndim_d);
        // Zi
        for (int doc_id=0; doc_id<num_docs; ++doc_id) {
            _cstm->update_Zi(doc_id);
        }
        assert(_sum_word_frequency.size() == _dataset.size());
        assert(_vocab->num_words() == _word_frequency.size());
        _old_alpha_words = new double[num_docs];
        _Zi_cache = new double[num_docs];
        // random sampling of words
        for (id word_id=0; word_id<vocabulary_size; ++word_id) {
            _num_updates_word[word_id] = 0;
            int count = _cstm->get_word_count(word_id);
            if (count <= _cstm->get_ignore_word_count()) {
                _ignored_vocabulary_size += 1;
                continue;
            }
            _random_word_ids.push_back(word_id);
        }
        std::shuffle(_random_word_ids.begin(), _random_word_ids.end(), sampler::mt);
        std::shuffle(_random_doc_ids_for_semantic.begin(), _random_doc_ids_for_semantic.end(), sampler::mt);
    }
    int add_document(string filepath) {
        wifstream ifs(filepath.c_str());
        assert(ifs.fail() == false);
        // add document
        int doc_id = _dataset.size();
        _dataset.push_back(vector<vector<id>>());
        _word_ids_in_doc.push_back(unordered_set<id>());
        _sum_word_frequency.push_back(0);
        // read file
        wstring sentence;
        vector<wstring> sentences;
        while (getline(ifs, sentence) && !ifs.eof()) {
            sentences.push_back(sentence);
        }
        for (wstring &sentence : sentences) {
            vector<wstring> words;
            split_word_by(sentence, L' ', words);
            add_sentence_to_doc(words, doc_id);
        }
        vector<string> components;
        split_string_by(filepath, '/', components);
        reverse(components.begin(), components.end());
        string name = components[0], author = components[1];
        string filename = author + "_" + name;
        _doc_filename_to_id[filename] = doc_id;
        _doc_id_to_filename[doc_id] = filename;
        return doc_id;
    }
    void add_sentence_to_doc(vector<wstring> &words, int doc_id) {
        if (words.size() > 0) {
            vector<vector<id>> &dataset = _dataset[doc_id];
            _sum_word_frequency[doc_id] += words.size();
            vector<id> word_ids;
            for (auto word : words) {
                if (word.size() == 0) {
                    continue;
                }
                id word_id = _vocab->add_string(word);
                word_ids.push_back(word_id);
                unordered_set<int> &docs = _docs_containing_word[word_id];
                docs.insert(doc_id);
                unordered_set<id> &word_set = _word_ids_in_doc[doc_id];
                word_set.insert(word_id);
                _word_frequency[word_id] += 1;
            }
            dataset.push_back(word_ids);
        }
    }
    // for validation dataset
    int add_validation_document(string filepath) {
        wifstream ifs(filepath.c_str());
        assert(ifs.fail() == false);
        // add document
        int doc_id = _validation_dataset.size();
        _validation_dataset.push_back(vector<vector<id>>());
        _word_ids_in_doc_validation.push_back(unordered_set<id>());
        _sum_word_frequency_validation.push_back(0);
        // read file
        wstring sentence;
        vector<wstring> sentences;
        while (getline(ifs, sentence) && !ifs.eof()) {
            sentences.push_back(sentence);
        }
        for (wstring &sentence : sentences) {
            vector<wstring> words;
            split_word_by(sentence, L' ', words);
            add_sentence_to_doc_validation(words, doc_id);
        }
        return doc_id;
    }
    // for validation dataset
    void add_sentence_to_doc_validation(vector<wstring> &words, int doc_id) {
        if (words.size() > 0) {
            vector<vector<id>> &dataset = _validation_dataset[doc_id];
            _sum_word_frequency_validation[doc_id] += words.size();
            vector<id> word_ids;
            for (auto word : words) {
                if (word.size() == 0) {
                    continue;
                }
                // need to dismiss unknown words
                if (!(_vocab->word_exists(word))) {
                    continue;
                }
                id word_id = _vocab->get_word_id(word);
                word_ids.push_back(word_id);
                unordered_set<id> &word_set = _word_ids_in_doc_validation[doc_id];
                word_set.insert(word_id);
            }
            dataset.push_back(word_ids);
        }
    }
    bool set_semantic_vector(wstring word, vector<double> vec) {
        if (!_vocab->word_exists(word)) return false;
        id word_id = _vocab->get_word_id(word);
        double* ptr = &vec[0];
        _cstm->set_semantic_vector(word_id, ptr);
        return true;
    }
    bool is_doc_contain_word(int doc_id, id word_id) {
        unordered_set<int> &set = _docs_containing_word[word_id];
        auto itr = set.find(doc_id);
        return itr != set.end();
    }
    int get_num_documents() {
        return _dataset.size();
    }
    int get_vocabulary_size() {
        return _word_frequency.size();
    }
    int get_ignored_vocabulary_size() {
        return _ignored_vocabulary_size;
    }
    int get_ndim_d() {
        return _cstm->_ndim_d;
    }
    int get_sum_word_frequency() {
        return std::accumulate(_sum_word_frequency.begin(), _sum_word_frequency.end(), 0);
    }
    // for validation dataset
    int get_sum_word_frequency_validation() {
        return std::accumulate(_sum_word_frequency_validation.begin(), _sum_word_frequency_validation.end(), 0);
    }
    int get_actual_sum_word_frequency() {
        int sum = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            for (id word_id=0; word_id<get_vocabulary_size(); ++word_id) {
                int count = _cstm->get_word_count_in_doc(word_id, doc_id);
                if (count <= _cstm->get_ignore_word_count()) {
                    sum += count;
                }
            }
        }
        return get_sum_word_frequency() - sum;
    }
    int get_actual_sum_word_frequency_validation() {
        int sum = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            for (id word_id=0; word_id<get_vocabulary_size(); ++word_id) {
                int count = _cstm->get_word_count_in_validation_doc(word_id, doc_id);
                if (count <= _cstm->get_ignore_word_count()) {
                    sum += count;
                }
            }
        }
        return get_sum_word_frequency_validation() - sum;
    }
    int get_num_word_vec_sampled() {
        return _num_word_vec_sampled;
    }
    int get_num_doc_vec_in_semantic_space_sampled() {
        return _num_doc_vec_in_semantic_space_sampled;
    }
    double get_alpha0() {
        return _cstm->_alpha0;
    }
    double get_mh_acceptance_rate_for_doc_vector_in_semantic_space() {
        return _num_acceptance_doc_in_semantic_space / (double)(_num_acceptance_doc_in_semantic_space + _num_rejection_doc_in_semantic_space);
    }
    double get_mh_acceptance_rate_for_word_vector() {
        return _num_acceptance_word / (double)(_num_acceptance_word + _num_rejection_word);
    }
    double get_mh_acceptance_rate_for_alpha0() {
        return _num_acceptance_alpha0 / (double)(_num_acceptance_alpha0 + _num_rejection_alpha0);
    }
    double *get_semantic_vector(id word_id) {
        double *old_vec = _cstm->get_semantic_vector(word_id);
        std::memcpy(_old_vec_copy, old_vec, _cstm->_ndim_d * sizeof(double));
        return _old_vec_copy;
    }
    double *get_doc_vector_in_semantic_space(int doc_id) {
        double *old_vec = _cstm->get_doc_vector_in_semantic_space(doc_id);
        std::memcpy(_old_vec_copy, old_vec, _cstm->_ndim_d * sizeof(double));
        return _old_vec_copy;
    }
    double *draw_semantic_vector(double *old_vec) {
        double *new_vec = _cstm->draw_semantic_vector(old_vec);
        std::memcpy(_new_vec_copy, new_vec, _cstm->_ndim_d * sizeof(double));
        return _new_vec_copy;
    }
    double *draw_doc_vector_in_semantic_space(double *old_vec) {
        double *new_vec = _cstm->draw_doc_vector_in_semantic_space(old_vec);
        std::memcpy(_new_vec_copy, new_vec, _cstm->_ndim_d * sizeof(double));
        return _new_vec_copy;
    }
    void set_num_threads(int num_threads) {
        _num_threads = num_threads;
    }
    void set_ignore_word_count(int count) {
        _cstm->set_ignore_word_count(count);
    }
    void set_ndim_d(int ndim_d) {
        _ndim_d = ndim_d;
    }
    void set_alpha0(double alpha0) {
        _cstm->_alpha0 = alpha0;
    }
    void set_sigma_u(double sigma_u) {
        _cstm->_sigma_u = sigma_u;
    }
    void set_sigma_v(double sigma_v) {
        _cstm->_sigma_v = sigma_v;
    }
    void set_sigma_phi(double sigma_phi) {
        _cstm->_sigma_phi = sigma_phi;
    }
    void set_sigma_alpha0(double sigma_alpha0) {
        _cstm->_sigma_alpha0 = sigma_alpha0;
    }
    void set_gamma_alpha_a(double gamma_alpha_a) {
        _cstm->_gamma_alpha_a = gamma_alpha_a;
    }
    void set_gamma_alpha_b(double gamma_alpha_b) {
        _cstm->_gamma_alpha_b = gamma_alpha_b;
    }
    double compute_log_likelihood_data() {
        double log_pw = 0;
        int n = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            unordered_set<id> &word_ids = _word_ids_in_doc[doc_id];
            log_pw += _cstm->compute_log_probability_document_given_words(doc_id, word_ids);
        }
        return log_pw;
    }
    // for validation dataset
    double compute_log_likelihood_validation_data() {
        double log_pw = 0;
        int n = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            unordered_set<id> &word_ids = _word_ids_in_doc_validation[doc_id];
            log_pw += _cstm->compute_log_probability_validation_document_given_words(doc_id, word_ids);
        }
        return log_pw;
    }
    double compute_perplexity() {
        double log_pw = 0;
        int n = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            unordered_set<id> &word_ids = _word_ids_in_doc[doc_id];
            log_pw += _cstm->compute_log_probability_document_given_words(doc_id, word_ids);
        }
        return cstm::exp(-log_pw / get_actual_sum_word_frequency());
    }
    // for validation dataset
    double compute_validation_perplexity() {
        double log_pw = 0;
        int n = 0;
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            unordered_set<id> &word_ids = _word_ids_in_doc_validation[doc_id];
            log_pw += _cstm->compute_log_probability_validation_document_given_words(doc_id, word_ids);
        }
        return cstm::exp(-log_pw / get_actual_sum_word_frequency_validation());
    }
    void update_all_Zi() {
        for (int doc_id=0; doc_id<get_num_documents(); ++doc_id) {
            _cstm->update_Zi(doc_id);
        }
    }
    // update document vector in semantic space
    void perform_mh_sampling_document_vector_in_semantic_space() {
        // choose doc vector for update
        if (_random_sampling_doc_vec_in_semantic_space_index + _num_threads >= _random_doc_ids_for_semantic.size()) {
            std::shuffle(_random_doc_ids_for_semantic.begin(), _random_doc_ids_for_semantic.end(), sampler::mt);
            _random_sampling_doc_vec_in_semantic_space_index = 0;
        }
        if (_num_threads == 1) {
            int doc_id = _random_doc_ids_for_semantic[_random_sampling_doc_vec_in_semantic_space_index];
            double *old_vec = get_doc_vector_in_semantic_space(doc_id);
            double *new_vec = draw_doc_vector_in_semantic_space(old_vec);
            accept_document_vector_in_semantic_space_if_needed(new_vec, old_vec, doc_id);
            _num_doc_vec_in_semantic_space_sampled += 1;
            _num_updates_doc_vec_in_semantic_space[doc_id] += 1;
            _random_sampling_doc_vec_in_semantic_space_index += 1;
            return;
        }
        // multi-thread
        for (int i=0; i<_num_threads; ++i) {
            int doc_id = _random_doc_ids_for_semantic[_random_sampling_doc_vec_in_semantic_space_index + i];
            double *old_vec = _cstm->get_doc_vector_in_semantic_space(doc_id);
            std::memcpy(_old_vec_copy_thread[i], old_vec, _cstm->_ndim_d * sizeof(double));
            double *new_vec = _cstm->draw_semantic_vector(old_vec);
            std::memcpy(_new_vec_copy_thread[i], new_vec, _cstm->_ndim_d * sizeof(double));
        }
        for (int i=0; i<_num_threads; ++i) {
            _semantic_doc_threads[i] = std::thread(&CSTMTrainer::worker_accept_semantic_document_vector_if_needed, this, i);
        }
        for (int i=0; i<_num_threads; ++i) {
            _semantic_doc_threads[i].join();
        }
        _random_sampling_doc_vec_in_semantic_space_index += _num_threads;

    }
    void worker_accept_semantic_document_vector_if_needed(int thread_id) {
        int doc_id = _random_doc_ids_for_semantic[_random_sampling_doc_vec_in_semantic_space_index + thread_id];
        double* old_vec = _old_vec_copy_thread[thread_id];
        double* new_vec = _new_vec_copy_thread[thread_id];
        accept_document_vector_in_semantic_space_if_needed(new_vec, old_vec, doc_id);
        _num_doc_vec_in_semantic_space_sampled += 1;
        _num_updates_doc_vec_in_semantic_space[doc_id] += 1;
    }
    bool accept_document_vector_in_semantic_space_if_needed(double *new_doc_vec, double *old_doc_vec, int doc_id) {
        double original_Zi = _cstm->get_Zi(doc_id);
        // likelihood of old doc vector
        double log_pw_old = _cstm->compute_log_probability_document(doc_id);
        // likelihood of new doc vector
        _cstm->set_doc_vector_in_semantic_space(doc_id, new_doc_vec);
        _cstm->update_Zi(doc_id);
        double log_pw_new = _cstm->compute_log_probability_document(doc_id);
        // assert(log_pw_old != 0);
        // assert(log_pw_new != 0);
        // prior distribution
        double log_prior_old = _cstm->compute_log_prior_vector(old_doc_vec);
        double log_prior_new = _cstm->compute_log_prior_vector(new_doc_vec);
        // acceptance rate
        double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
        double acceptance_ratio = std::min(1.0, cstm::exp(log_acceptance_rate));
        double bernoulli = sampler::uniform(0, 1);
        if (bernoulli <= acceptance_ratio) {
            _num_acceptance_doc_in_semantic_space += 1;
            return true;
        }
        // undo
        _cstm->set_doc_vector_in_semantic_space(doc_id, old_doc_vec);
        _cstm->set_Zi(doc_id, original_Zi);
        _num_rejection_doc_in_semantic_space += 1;
        return false;
    }
    void perform_mh_sampling_alpha0() {
        int doc_id = sampler::uniform_int(0, _cstm->_num_documents - 1);
        double old_alpha0 = _cstm->get_alpha0();
        double new_alpha0 = _cstm->draw_alpha0(old_alpha0);
        accept_alpha0_if_needed(new_alpha0, old_alpha0);
    }
    bool accept_alpha0_if_needed(double new_alpha0, double old_alpha0) {
        int num_docs = _dataset.size();
        // likelihood of old a0
        double log_pw_old = 0;
        for (int doc_id=0; doc_id<num_docs; ++doc_id) {
            log_pw_old += _cstm->compute_log_probability_document(doc_id);
            _Zi_cache[doc_id] = _cstm->get_Zi(doc_id);
        }
        // likelihood of new a0
        _cstm->set_alpha0(new_alpha0);
        update_all_Zi();
        double log_pw_new = 0;
        for (int doc_id=0; doc_id<num_docs; ++doc_id) {
            log_pw_new += _cstm->compute_log_probability_document(doc_id);
        }
        // prior distribution
        double log_prior_old = _cstm->compute_log_prior_alpha0(old_alpha0);
        double log_prior_new = _cstm->compute_log_prior_alpha0(new_alpha0);
        // acceptance rate
        double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
        double acceptance_ratio = std::min(1.0, cstm::exp(log_acceptance_rate));
        double bernoulli = sampler::uniform(0, 1);
        if (bernoulli <= acceptance_ratio) {
            _num_acceptance_alpha0 += 1;
            return true;
        }
        _num_rejection_alpha0 += 1;
        // undo
        _cstm->set_alpha0(old_alpha0);
        for (int doc_id=0; doc_id<num_docs; ++doc_id) {
            _cstm->set_Zi(doc_id, _Zi_cache[doc_id]);
        }
        return false;
    }
    void save(string filename) {
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oarchive(ofs);
        oarchive << *_vocab;
        oarchive << *_cstm;
        oarchive << _word_frequency;
        oarchive << _word_ids_in_doc;
        oarchive << _docs_containing_word;
        oarchive << _sum_word_frequency;
        oarchive << _doc_filename_to_id;
        oarchive << _doc_id_to_filename;
    }
    bool load(string filename) {
        std::ifstream ifs(filename);
        if (ifs.good()) {
            _vocab = new Vocab();
            _cstm = new CSTM();
            boost::archive::binary_iarchive iarchive(ifs);
            iarchive >> *_vocab;
            iarchive >> *_cstm;
            iarchive >> _word_frequency;
            iarchive >> _word_ids_in_doc;
            iarchive >> _docs_containing_word;
            iarchive >> _sum_word_frequency;
            iarchive >> _doc_filename_to_id;
            iarchive >> _doc_id_to_filename;
            return true;
        }
        return false;
    }
};

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

void normalize_vector(vector<vector<double>> &vec, double tau) {
    vector<double> &tmp = vec[0];
    int d = (int)tmp.size();
    // calculate mean vector of word embeddings and execute centering
    vector<double> mean_vec(d, 0);
    for (int k=0; k<vec.size(); ++k) {
        vector<double> &tar = vec[k];
        for (int i=0; i<tar.size(); ++i) {
            mean_vec[i] += tar[i];
        }
    }
    for (int i=0; i<mean_vec.size(); ++i) {
        mean_vec[i] /= (double)vec.size();
    }
    // for all vector
    for (int k=0; k<vec.size(); ++k) {
        vector<double> &tar = vec[k];
        for (int i=0; i<tar.size(); ++i) {
            tar[i] -= mean_vec[i];
        }
    }
    // calculate V^{-1} \sum_k (\phi(w_k)^T \phi(w_k))
    double Einner = 0;
    for (int k=0; k<vec.size(); ++k) {
        vector<double> &tar = vec[k];
        double inner = 0;
        for (int i=0; i<tar.size(); ++i) {
            inner += tar[i] * tar[i];
        }
        Einner += inner;
    }
    Einner /= (double)vec.size();
    // scaling each element with square root of expected value of inner product
    double sqrt_inner = std::sqrt(Einner);
    for (int k=0; k<vec.size(); ++k) {
        vector<double> &tar = vec[k];
        for (int i=0; i<tar.size(); ++i) {
            tar[i] /= sqrt_inner;
            // tar[i] /= tau;
        }
    }
}

void read_aozora_data(string data_path, CSTMTrainer &trainer) {
    const char* path = data_path.c_str();
    DIR *dp_author; dp_author = opendir(path);
    assert (dp_author != NULL);
    dirent* entry_author = readdir(dp_author);
    while (entry_author != NULL){
        const char *cstr = entry_author->d_name;
        string author = string(cstr);
        if (author == ".." || author == ".") {
            entry_author = readdir(dp_author);
            continue;
        }
        string tmp = data_path + author;
        const char* path_to_file = tmp.c_str();
        DIR *dp_file; dp_file = opendir(path_to_file);
        assert(dp_file != NULL);
        dirent* entry_file = readdir(dp_file);
        while (entry_file != NULL) {
            const char *cstr2 = entry_file->d_name;
            string file_path = string(cstr2);
            if (ends_with(file_path, ".txt")) {
                // std::cout << "loading " << file_path << std::endl;
                int doc_id = trainer.add_document(data_path + author + "/" + file_path);
            }
            entry_file = readdir(dp_file);
        }
        entry_author = readdir(dp_author);
    }
}

void read_data(string data_path, CSTMTrainer &trainer) {
    const char* path = data_path.c_str();
    DIR *dp;
    dp = opendir(path);
    assert (dp != NULL);
    dirent* entry = readdir(dp);
    while (entry != NULL){
        const char *cstr = entry->d_name;
        string file_path = string(cstr);
        if (ends_with(file_path, ".txt")) {
            // std::cout << "loading " << file_path << std::endl;
            int doc_id = trainer.add_document(data_path + file_path);
        }
        entry = readdir(dp);
    }
}

void read_validation_data(string data_path, CSTMTrainer &trainer) {
    const char* path = data_path.c_str();
    DIR *dp;
    dp = opendir(path);
    assert (dp != NULL);
    dirent* entry = readdir(dp);
    while (entry != NULL){
        const char *cstr = entry->d_name;
        string file_path = string(cstr);
        if (ends_with(file_path, ".txt")) {
            // std::cout << "loading " << file_path << std::endl;
            int doc_id = trainer.add_validation_document(data_path + file_path);
        }
        entry = readdir(dp);
    }
}

// hyper parameters flags
DEFINE_int32(ndim_d, 20, "number of hidden size");
DEFINE_double(sigma_u, 0.01, "params: sigma_u");
DEFINE_double(sigma_v, 0.01, "params: sigma_v");
DEFINE_double(sigma_phi, 0.02, "params: sigma_phi");
DEFINE_double(sigma_alpha0, 0.2, "params: sigma_alpha0");
DEFINE_int32(gamma_alpha_a, 5, "params: gamma_alpha_a");
DEFINE_int32(gamma_alpha_b, 500, "params: gamma_alpha_b");
DEFINE_int32(ignore_word_count, 4, "number of ignore word");    // minimum_word_count = `ignore_word_count` + 1 
DEFINE_int32(epoch, 100, "num of epoch");
DEFINE_int32(num_threads, 1, "num of threads");
DEFINE_string(data_path, "../data/train/", "directory train data located");
DEFINE_string(validation_data_path, "null", "directory validation data located");
DEFINE_string(model_path, "./cstm.model", "saveplace of model");
DEFINE_string(vec_path, "./vec.bin", "saveplace of pre-trained word vector");

void train(int argc, char *argv[]) {
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    // load pre-trained vectors
    vector<wstring> vocab;
    vector<vector<double>> semantic_vec;
    load_vector(FLAGS_vec_path, vocab, semantic_vec);
    // normalize word vector
    double tau = std::sqrt(FLAGS_ndim_d);   // params; we simply set tau = \sqrt(d)
    normalize_vector(semantic_vec, tau);
    CSTMTrainer trainer;
    // set hyper parameter
    trainer.set_ndim_d(FLAGS_ndim_d);
    trainer.set_sigma_u(FLAGS_sigma_u);
    trainer.set_sigma_v(FLAGS_sigma_v);
    trainer.set_sigma_phi(FLAGS_sigma_phi);
    trainer.set_sigma_alpha0(FLAGS_sigma_alpha0);
    trainer.set_gamma_alpha_a(FLAGS_gamma_alpha_a);
    trainer.set_gamma_alpha_b(FLAGS_gamma_alpha_b);
    trainer.set_ignore_word_count(FLAGS_ignore_word_count);
    trainer.set_num_threads(FLAGS_num_threads);
    bool validation = (FLAGS_validation_data_path != "null");
    // read file
    // read_aozora_data(FLAGS_data_path, trainer);
    read_data(FLAGS_data_path, trainer);
    if (validation) {
        read_validation_data(FLAGS_validation_data_path, trainer);
    }
    // prepare model
    trainer.prepare(validation=validation);
    assert(trainer._ndim_d == semantic_vec[0].size());
    // set pre-trained semantic vectors
    int added_word_count = 0;
    for (int i=0; i<vocab.size(); ++i) {
        bool res = trainer.set_semantic_vector(vocab[i], semantic_vec[i]);
        added_word_count += (int)(res);
    }
    std::cout << "count of obtained word vector: " << added_word_count << std::endl;
    // summary
    std::cout << "vocabulary size: " << trainer.get_vocabulary_size() << std::endl;
    std::cout << "ignored vocabulary size: " << trainer.get_ignored_vocabulary_size() << std::endl;
    std::cout << "actual vocabulary size: " << trainer.get_vocabulary_size() - trainer.get_ignored_vocabulary_size() << std::endl;
    std::cout << "num of documents: " << trainer.get_num_documents() << std::endl;
    std::cout << "num of words: " << trainer.get_sum_word_frequency() << std:: endl;
    std::cout << "dimension size of latent space: " << trainer.get_ndim_d() << std::endl;
    // training
    int iter = 10000;
    for (int i=0; i<FLAGS_epoch; ++i) {
        for (int j=0; j<iter; ++j) {
            trainer.perform_mh_sampling_document_vector_in_semantic_space();
            if (j % (iter/10) == 0) {
                trainer.perform_mh_sampling_alpha0();
            }
        }
        std::cout << "epoch " << i+1 << "/" << FLAGS_epoch << std::endl;
        // logging temporary result
        std::cout << "perplexity: " << trainer.compute_perplexity() << std::endl;
        std::cout << "log likelihood: " << trainer.compute_log_likelihood_data() << std::endl;
        // logging score for validation dataset if validation == true
        if (validation) {
            std::cout << "validation perplexity: " << trainer.compute_validation_perplexity() << std::endl;
            std::cout << "validation log likelihood: " << trainer.compute_log_likelihood_validation_data() << std::endl;
        }
        // logging statistics
        std::cout << "MH acceptance:" << std::endl;
        std::cout << "    semantic_doc: " << trainer.get_mh_acceptance_rate_for_doc_vector_in_semantic_space() << std::endl;
        std::cout << "    alpha0: " << trainer.get_mh_acceptance_rate_for_alpha0() << std::endl;
        trainer.save(FLAGS_model_path);
        trainer.reset_statistics();
    }
}

int main(int argc, char *argv[]) {
    train(argc, argv);
    return 0;
}