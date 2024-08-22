#ifndef VECTOR_SETTINGS_HPP
#define VECTOR_SETTINGS_HPP

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <tuple>

namespace vector {
/// @brief Literal string as NNTP
///
/// @tparam N length of string
template <std::size_t N> struct literal {
    char data[N];

    constexpr literal(const char (&literal)[N]) { std::copy_n(literal, N, data); }

    template <std::size_t M>
    constexpr bool operator==(this const literal<N> &self, const literal<M> &other) {
        if constexpr (M != N)
            return false;

        for (const auto i : std::ranges::iota_view{0uz, N}) {
            if (self.data[i] != other.data[i])
                return false;
        }

        return true;
    }
};

// Settings for for_each function
enum SimdSettings {
	/// Disables SIMD, will use a regular for-loop
    NONE,
	/// Enables SIMD via OpenMP (requires OpenMP)
    SIMD,
	/// Unroll loops
    UNROLL,
};

/**
 * @brief Represents a (key, value) pair for the settings
 *
 * @tparam _key The name of the settings
 * @tparam _value The value of the settings
 */
template <literal _key, SimdSettings _value> struct SettingsField {
    constexpr static inline literal key = _key;
    constexpr static inline SimdSettings value = _value;
};

/// @cond
template <class...> class SettingsRegistry;
/// @endcond

/**
 * @brief Registry holding vector::SimdSettings for the vector's dispatch policy
 *
 * @tparam SettingsField A (key, value) pair for the setting's name and value
 * \code{.cpp}
 * SettingsField<"default", SimdSettings::SIMD>, // required
 * SettingsField<"add(this Self& self, const Other& other)", SimdSettings::UNROLL>, // unroll for add
 * ...
 * \endcode
 */
template <literal... Names, SimdSettings... Settings>
class SettingsRegistry<SettingsField<Names, Settings>...> {
    using settings = std::tuple<SettingsField<Names, Settings>...>;

    template <std::size_t I, literal key> static consteval decltype(auto) get_impl() {
        using elem = std::tuple_element_t<I, settings>;
        if constexpr (elem::key == key) {
            return (elem::value);
        } else if constexpr (I + 1 < std::tuple_size_v<settings>) {
            return get_impl<I + 1, key>();
        } else {
            if constexpr (key == literal<8>{"default"}) {
                // No default key
                []<bool v = true>() { static_assert(v, "No default key for settings"); }();
            } else {
                // Get default key if not found for current key
                return get_impl<0, "default">();
            }
        }
    }

  public:
    template <literal key> static consteval decltype(auto) get() { return get_impl<0, key>(); }
};
} // namespace vector

#endif // VECTOR_SETTINGS_HPP
