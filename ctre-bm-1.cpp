#include <ctre.hpp>
#include <cstring>

#if 0
static constexpr inline ctll::fixed_string pattern = "([0-9]++),([a-z]++)";

bool match(std::string_view sv) noexcept {
#if __cpp_nontype_template_parameter_class
    return ctre::match<"([0-9]++),([a-z]++)">(sv);
#else
	return ctre::match<pattern>(sv);
#endif
}
#endif

template <std::size_t N, const char (&rx)[N]>
struct matcher {
  static constexpr ctll::fixed_string<N> pat = rx;
  static bool match(std::string_view sv) {
    return ctre::match<pat>(sv);
  }
};

template <std::size_t N, const char (&rx)[N]>
bool matchs(std::string_view sv) noexcept {
  return matcher<N, rx>::match(sv);
}

template <std::size_t N, const char (&rx)[N]>
[[clang::jit]] bool match(std::string_view sv) noexcept {
  return matcher<N, rx>::match(sv);
}

static constexpr char p[] = "([0-9]++),([a-z]++)";
int main(int argc, char *argv[]) {
  constexpr ctll::fixed_string ps(p);

  matchs<sizeof(p), p>(argv[0]);

  return !match<std::strlen(argv[argc-1])+1, argv[argc-1]>(argv[0]);
}

