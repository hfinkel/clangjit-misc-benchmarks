#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>

#include <chrono>

#include <ctre.hpp>

#ifndef CTRE_ONLY
#include <boost/regex.hpp>
#include <boost/xpressive/xpressive.hpp>

extern "C" {
#define PCRE2_CODE_UNIT_WIDTH 8
#define PCRE2_STATIC
#include <pcre2.h>
}
#endif

namespace chrono = std::chrono;

#ifndef CTRE_ONLY
bool test_boost(const char *pattern, std::string_view sv, int rep) {
  auto re = boost::regex(pattern);
      
  bool m = false;

  auto start = chrono::system_clock::now();

  for (int i = 0; i < rep; ++i)
    m |= regex_search(sv.begin(), sv.end(), re);

  auto end = chrono::system_clock::now();
  std::cout << "Boost Regex: " << std::chrono::duration<double>(end - start).count() << " s\n";

  return m;
}

bool test_xpressive(const char *pattern, std::string_view sv, int rep) {
  using svregex = boost::xpressive::basic_regex<std::string_view::const_iterator>;
  svregex re = svregex::compile(pattern);
        
  bool m = false;

  auto start = chrono::system_clock::now();

  for (int i = 0; i < rep; ++i)
    m |= bool(boost::xpressive::regex_search(sv, re));

  auto end = chrono::system_clock::now();
  std::cout << "Boost Xpressive (dynamic): " << std::chrono::duration<double>(end - start).count() << " s\n";

  return m;
}

bool test_pcre(const char *pat, std::string_view sv, int rep) {
  int errornumber = 0;
  size_t erroroffset = 0;
  const auto * pattern = reinterpret_cast<const unsigned char *>(pat);
  pcre2_code * re = pcre2_compile(pattern, PCRE2_ZERO_TERMINATED, 0, &errornumber, &erroroffset, NULL);
        
  if (!re) {
    std::cerr << "compilation failed\n";
    return false;
  }
        
  pcre2_match_context * mcontext = pcre2_match_context_create(NULL);
  auto match_data = pcre2_match_data_create_from_pattern(re, NULL);
        
  bool m = false;

  auto start = chrono::system_clock::now();

  for (int i = 0; i < rep; ++i)
    m |= pcre2_match(re, reinterpret_cast<const unsigned char *>(sv.data()), sv.length(), 0, 0, match_data, mcontext) >= 0;

  auto end = chrono::system_clock::now();
  std::cout << "PCRE: " << std::chrono::duration<double>(end - start).count() << " s\n";

  pcre2_code_free(re);

  return m;
}

bool test_pcre_jit(const char *pat, std::string_view sv, int rep) {
  int errornumber = 0;
  size_t erroroffset = 0;
  const auto * pattern = reinterpret_cast<const unsigned char *>(pat);
  pcre2_code * re = pcre2_compile(pattern, PCRE2_ZERO_TERMINATED, 0, &errornumber, &erroroffset, NULL);
        
  if (!re) {
    std::cerr << "compilation failed\n";
    return false;
  }
        
  pcre2_match_context * mcontext = pcre2_match_context_create(NULL);
  auto match_data = pcre2_match_data_create_from_pattern(re, NULL);
  
  pcre2_jit_compile(re, PCRE2_JIT_COMPLETE);
  pcre2_jit_stack * jit_stack = pcre2_jit_stack_create(32*1024, 512*1024, NULL);
  pcre2_jit_stack_assign(mcontext, NULL, jit_stack);
 
  bool m = false;

  auto start = chrono::system_clock::now();

  for (int i = 0; i < rep; ++i)
    m |= pcre2_match(re, reinterpret_cast<const unsigned char *>(sv.data()), sv.length(), 0, 0, match_data, mcontext) >= 0;

  auto end = chrono::system_clock::now();
  std::cout << "PCRE (JIT): " << std::chrono::duration<double>(end - start).count() << " s\n";

  pcre2_code_free(re);

  return m;
}
#endif

template <std::size_t N, const char (&rx)[N]>
struct matcher {
  static constexpr ctll::fixed_string<N> pat = rx;
  static bool match(std::string_view sv, int rep) {
    bool m = false;

    auto start = chrono::system_clock::now();

    for (int i = 0; i < rep; ++i)
      m |= ctre::match<pat>(sv);

    auto end = chrono::system_clock::now();
    std::cout << "JIT CTRE: " << std::chrono::duration<double>(end - start).count() << " s\n";

    return m;
  }
};

template <std::size_t N, const char (&rx)[N]>
bool matchs(std::string_view sv, int rep) noexcept {
  return matcher<N, rx>::match(sv, rep);
}

template <std::size_t N, const char (&rx)[N]>
[[clang::jit]] bool match(std::string_view sv, int rep) noexcept {
  return matcher<N, rx>::match(sv, rep);
}

int main(int argc, char *argv[]) {
  std::string pattern("a*b*");
  if (argc > 1)
    pattern = argv[1];

  std::string fill("a");
  if (argc > 2)
    fill = argv[2];

  unsigned repl = 16;
  if (argc > 3)
    repl = std::atoi(argv[3]);

  unsigned rep = 1024;
  if (argc > 4)
    rep = std::atoi(argv[4]);

  while (--repl)
    fill = fill + fill;

#ifndef CTRE_ONLY
  test_boost(pattern.c_str(), fill, rep);
  test_xpressive(pattern.c_str(), fill, rep);

  test_pcre(pattern.c_str(), fill, rep);
  test_pcre_jit(pattern.c_str(), fill, rep);
#endif

  (void) match<pattern.size()+1, pattern.c_str()>(fill, rep);

  return 0;
}

