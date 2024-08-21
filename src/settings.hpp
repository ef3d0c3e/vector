#ifndef VECTOR_SETTINGS_HPP
#define VECTOR_SETTINGS_HPP

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <tuple>

namespace vector
{
template <std::size_t N>
struct literal
{
	char data[N];

	constexpr literal(const char (&literal)[N])
	{
		std::copy_n(literal, N, data);
	}

	template <std::size_t M>
	constexpr bool operator==(this const literal<N>& self, const literal<M>& other)
	{
		if constexpr (M != N)
			return false;

		for (const auto i : std::ranges::iota_view{ 0uz, N })
		{
			if (self.data[i] != other.data[i])
				return false;
		}

		return true;
	}

};

// Settings for for_each function
struct SimdSettings
{
	/// Enables simd globally
	bool enabled = true;

	/// Unroll for loops if `N < max_unroll`
	std::size_t max_unroll = 0;
};

/**
 * @brief Represents a (key, value) pair for the settings
 * 
 * @tparam _key The name of the settings
 * @tparam _value The value of the settings
 */
template <literal _key, SimdSettings _value>
struct SettingsField
{
	constexpr static inline literal key = _key;
	constexpr static inline SimdSettings value = _value;
};

/**
 * @brief The registry holding the custom settings for SIMD
 */
template <class...>
struct SettingsRegistry;

template <literal... Names, SimdSettings... Settings>
class SettingsRegistry<SettingsField<Names, Settings>...>
{
	using settings = std::tuple<SettingsField<Names, Settings>...>;

	template <std::size_t I, literal key>
	static constexpr decltype(auto) get_impl()
	{
		using elem = std::tuple_element_t<I, settings>;
		if constexpr (elem::key == key)
		{
			return (elem::value);
		}
		else if constexpr (I + 1 < std::tuple_size_v<settings>)
		{
			return get<I + 1, key>();
		}
		else
		{
			if constexpr (key == literal<8>{"default"})
			{
				// No default key
				[]<bool v = true>() {
					static_assert(v, "No default key for settings");
				}();
			}
			else
			{
				// Get default key if not found for current key
				return get_impl<0, "default">();
			}
		}
	}
public:

	template <literal key>
	static constexpr decltype(auto) get()
	{
		return get_impl<0, key>();
	}

};
} // vector

#endif // VECTOR_SETTINGS_HPP