#pragma once
#include <iostream>
#include <cmath>
#include "fmath.hpp"
#define NDIM_D 300              // 文書,単語ベクトルの次元数
// #define SCALE_U 1               // 文書ベクトルの事前分布の分ののスケーリング係数
// #define SCALE_V 1               // 文書ベクトルの事前分布の分ののスケーリング係数
#define SIGMA_U 0.01            // 意味空間上の文書ベクトののランダムウォーク幅
#define SIGMA_V 0.01            // スタイル空間上の文書ベクトルのランダムウォーク幅
#define SIGMA_PHI 0.02          // 単語ベクトルのランダムウォーク幅
#define SIGMA_ALPHA 0.2         // a0のランダムウォーク幅
#define GAMMA_ALPHA_A 5         // a0のガンマ事前分布のハイパーパラメータ
#define GAMMA_ALPHA_B 1         // a0のガンマ事前分布のハイパーパラメータ
using id = size_t;

namespace cstm {
    double exp(double x) {
        return std::exp(x);
    }
    double norm(double *a, int length) {
        double norm = 0;
        for (int i=0; i<length; ++i) {
            norm += a[i] * a[i];
        }
        return sqrt(norm);
    }
    double inner(double *a, double *b, int length) {
        double inner = 0;
        for (int i=0; i<length; ++i) {
            inner += a[i] * b[i];
        }
        return inner;
    }
    // do not use
    // correlation coefficient
    // actually, we have to normalize vector `a` which is not generated from gaussian
    double normalized_linear(double *a, double *b, int length) {
        double inner = 0;
        double a2 = 0, b2 = 0;
        for (int i=0; i<length; ++i) {
            inner += a[i] * b[i];
            a2 += a[i] * a[i];
            b2 += b[i] * b[i];
        }
        return inner / std::sqrt(a2 * b2);
    }
    void dump_vec(double *vec, int len) {
        std::cout << "[";
        for (int i=0; i<len-1; ++i) {
            std::cout << vec[i] << ", ";
        }
        std::cout << vec[len-1] << "]" << std::endl;
    }
}