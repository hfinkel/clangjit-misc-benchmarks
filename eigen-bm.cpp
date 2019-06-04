#ifdef __CLING__
#pragma cling optimize(3)
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

#include <Eigen/Core>

#ifdef __CLING__
#include <sstream>
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/Value.h"
#include "cling/Interpreter/Transaction.h"
#pragma cling optimize(3)
#endif

// A benchmark for ClangJIT in the style of:
// https://github.com/eigenteam/eigen-git-mirror/blob/master/bench/benchmark.cpp

using namespace std;
using namespace Eigen;

#ifndef __has_feature
  #define __has_feature(x) 0
#endif

#if __has_feature(clang_cxx_jit)
template <typename T, int size>
[[clang::jit]] void test_jit_sz(int repeat) {
  Matrix<T,size,size> I = Matrix<T,size,size>::Ones();
  Matrix<T,size,size> m;
  for(int i = 0; i < size; i++)
  for(int j = 0; j < size; j++) {
    m(i,j) = (i+size*j);
  }

  auto start = chrono::system_clock::now();

  for (int r = 0; r < repeat; ++r) {
    m = Matrix<T,size,size>::Ones() + T(0.00005) * (m + (m*m));
  }

  auto end = chrono::system_clock::now();
  cout << "JIT: " << std::chrono::duration<double>(end - start).count() << " s\n";
}

void test_jit(std::string &type, int size, int repeat) {
  return test_jit_sz<type, size>(repeat);
}
#else
void test_jit(std::string &type, int size, int repeat) {
  cout << "JIT not supported\n";
}
#endif

#ifdef __CLING__
template <typename T, int size>
void test_cling_sz(int repeat) {
  cout << "OL: " << gCling->getDefaultOptLevel() << "\n";
  Matrix<T,size,size> I = Matrix<T,size,size>::Ones();
  Matrix<T,size,size> m;
  for(int i = 0; i < size; i++)
  for(int j = 0; j < size; j++) {
    m(i,j) = (i+size*j);
  }

  auto start = chrono::system_clock::now();

  for (int r = 0; r < repeat; ++r) {
    m = Matrix<T,size,size>::Ones() + T(0.00005) * (m + (m*m));
  }

  auto end = chrono::system_clock::now();
  cout << "JIT: " << std::chrono::duration<double>(end - start).count() << " s\n";
}

void test_cling(const std::string &type, int size, int repeat) {
  std::stringstream ss;
  ss << "#pragma cling optimize(3)" << "\n";
  ss << "test_cling_sz<" << type << ", " << size << ">(" << repeat << ")";

  cling::Value V;
  gCling->evaluate(ss.str(), V); 
}
#endif

#if defined(SPEC_TYPE) && defined(SPEC_SIZE)
template <typename T, int size>
void test_spec_sz(int repeat) {
  Matrix<T,size,size> I = Matrix<T,size,size>::Ones();
  Matrix<T,size,size> m;
  for(int i = 0; i < size; i++)
  for(int j = 0; j < size; j++) {
    m(i,j) = (i+size*j);
  }

  auto start = chrono::system_clock::now();

  for (int r = 0; r < repeat; ++r) {
    m = Matrix<T,size,size>::Ones() + T(0.00005) * (m + (m*m));
  }

  auto end = chrono::system_clock::now();
  cout << "Spec: " << std::chrono::duration<double>(end - start).count() << " s\n";
}

void test_spec(std::string &type, int size, int repeat) {
  return test_spec_sz<SPEC_TYPE, SPEC_SIZE>(repeat);
}
#else
void test_spec(std::string &type, int size, int repeat) {
  cout << "Spec. not supported\n";
}
#endif

#ifndef NO_AOT
template <typename T>
void test_aot(int size, int repeat) {
  Matrix<T,Dynamic,Dynamic> I = Matrix<T,Dynamic,Dynamic>::Ones(size, size);
  Matrix<T,Dynamic,Dynamic> m(size, size);
  for(int i = 0; i < size; i++)
  for(int j = 0; j < size; j++) {
    m(i,j) = (i+size*j);
  }

  auto start = chrono::system_clock::now();

  for (int r = 0; r < repeat; ++r) {
    m = Matrix<T,Dynamic,Dynamic>::Ones(size, size) + T(0.00005) * (m + (m*m));
  }

  auto end = chrono::system_clock::now();
  cout << "AoT: " << std::chrono::duration<double>(end - start).count() << " s\n";
}

void test_aot(const std::string &type, int size, int repeat) {
  if (type == "float")
    test_aot<float>(size, repeat);
  else if (type == "double")
    test_aot<double>(size, repeat);
  else if (type == "long double")
    test_aot<long double>(size, repeat);
  else
    cout << type << "not supported for AoT\n";
}
#else
void test_aot(std::string &type, int size, int repeat) {
  cout << "AoT not supported\n";
}
#endif

int main(int argc, char *argv[]) {
  int repeat = 40000000;
  if (argc > 1)
    repeat = atoi(argv[1]);

  int size = 3;
  if (argc > 2)
    size = atoi(argv[2]);

  string type("double");
  if (argc > 3)
    type = argv[3];

  test_jit(type, size, repeat);
  test_spec(type, size, repeat);
  test_aot(type, size, repeat);

  return 0;
}

