#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>

#include <Eigen/Core>

// A benchmark for ClangJIT in the style of:
// https://github.com/eigenteam/eigen-git-mirror/blob/master/bench/benchmark.cpp

using namespace std;
using namespace Eigen;

#ifndef __has_feature
  #define __has_feature(x) 0
#endif

#if __has_feature(clang_cxx_jit)
template <typename T, int size>
__global__ void dokernel(int repeat, T *out) {
  Matrix<T,size,size> I = Matrix<T,size,size>::Ones();
  Matrix<T,size,size> m;
  for(int i = 0; i < size; i++)
  for(int j = 0; j < size; j++) {
    m(i,j) = (i+size*j);
  }

  for (int r = 0; r < repeat; ++r) {
    m = Matrix<T,size,size>::Ones() + T(0.00005) * (m + (m*m));
  }

  out[threadIdx.x] = m(0, 0);
}

template <typename T, int size>
[[clang::jit]] void test_jit_sz(int repeat) {
  auto start = chrono::system_clock::now();

  const int w = 1;
  T *harr = new T[w];
  memset(harr, 0, sizeof(T)*w);
  T *darr;
  cudaMalloc((void **) &darr, w*sizeof(T));
  cudaMemcpy(darr, harr, w*sizeof(T), cudaMemcpyHostToDevice);

  dokernel<T, size><<<1, w>>>(repeat, darr);

  cudaMemcpy(harr, darr, w*sizeof(T), cudaMemcpyDeviceToHost);

#if 0
  for (int i = 0; i < w; ++i)
    cout << "h" << i << ": " << harr[i] << "\n";
#endif

  auto end = chrono::system_clock::now();
  cout << "JIT: " << std::chrono::duration<double>(end - start).count() << " s\n";
}

void test_jit(std::string &type, int size, int repeat) {
  return test_jit_sz<type, size>(repeat);
}
#else
void test_jit(std::string &type, int size, int repeat) {
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

  return 0;
}

