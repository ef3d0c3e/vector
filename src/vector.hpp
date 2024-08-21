#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <concepts>
#include <fmt/format.h>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <iostream>

#include "settings.hpp"

namespace vector
{
template <template <class, std::size_t> class S, class T, std::size_t N>
concept VecStorage = requires(S<T, N>&& s, std::size_t i) {
	{ static_cast<S<T, N>&>(s)[i] } -> std::same_as<T&>;
	{ static_cast<const S<T, N>&>(s)[i] } -> std::same_as<const T&>;
};

/// Settings for the @see Vector class
struct VectorSettings
{
	struct Arithmetic
	{
		/// Enables arithmetic methods `add`, `mul`, etc.
		bool enabled = true;

		/// Overload operators for arithmetic methods (+, -, etc.)
		bool overloads = true;

		/// Overload assignment operators (+=, /=, etc.)
		bool assignment = true;

		/// Allows implicit type casting for operators
		/// Generally `Vector<float> + Vector<int>` is ill formed
		/// Enabling this settings will cast the result type to the type of the lhs
		bool implicit_casting = true;
	} arithmetic{};

	/// Whether the vector extends it's storage class
	/// Otherwise the storage is kept as a member
	bool extend_storage = true;
};

namespace details
{
/// Utility to validate vector settings.
/// Meant to be used to give an insight about incompatible settings
///
/// @tparam S The settings to validate
template <VectorSettings S>
consteval bool validate_settings()
{
	// Arithmetic
	static_assert(S.arithmetic.enabled || !S.arithmetic.overloads, "Cannnot enable arithmetic overloads if arithmetic is disabled");
	static_assert(S.arithmetic.overloads || !S.arithmetic.assignment, "Cannnot enable arithmetic assignment if arithmetic overloads are disabled");
	static_assert(S.arithmetic.overloads || !S.arithmetic.implicit_casting, "Cannnot enable arithmetic implicit casting if arithmetic overloads are disabled");

	return true;
}


template <class T, std::size_t N>
struct default_storage
{
	T data[N];

	constexpr inline decltype(auto) operator[](std::size_t i) noexcept
	{
		return (data[i]);
	}

	constexpr inline decltype(auto) operator[](std::size_t i) const noexcept
	{
		return (data[i]);
	}
};
static_assert(VecStorage<default_storage, float, 5>);

template <std::size_t N, SimdSettings simd, class F>
void for_each(F&& fn)
{
	if constexpr (N <= simd.max_unroll)
	{
		[&]<auto... i>(std::index_sequence<i...>) {
			((fn(i)), ...);
		}(std::make_index_sequence<N>{});
	}
	else
	{
		if constexpr (simd.enabled)
		{
			[[omp::directive(simd)]]
			for (std::size_t i = 0; i < N; ++i)
			{
				fn(i);
			}
		}
		else
		{
			for (std::size_t i = 0; i < N; ++i)
			{
				fn(i);
			}
		}
	}
}

template <class Left, class Right, class Op>
concept valid_operator = requires(const Left& l, const Right& r, Op&& op, std::size_t i) {
	typename Left::base_type;
	typename Right::base_type;
	{ op(l[i], r[i]) } -> std::same_as<typename Left::base_type>;
};

struct empty_t
{};

struct arithmetic_overloads
{
	template <class Self, class Other = Self>
		requires valid_operator<Self, Other, decltype([](auto const& a, auto const& b) -> decltype(auto) { return a + b; })>
	constexpr Self operator+(this const Self& self, const Other& other)
	{
		Self ret{};
		for_each<Self::size(), Self::settings.simd>([&](std::size_t i) {
			ret[i] = self[i] + other[i];
		});

		return ret;
	}

	/*
	template <class Self, class Other = Self>
		requires(Self::settings.arithmetic.assignment) &&
				valid_operator<Self, Other, decltype([](auto const& a, auto const& b) -> decltype(auto) { return a + b; })>
	constexpr Self& operator+=(this Self& self, const Other& other)
	{
		for_each<Self::size(), Self::settings.simd>([&](std::size_t i) {
			self[i] = self[i] + other[i];
		});
		return self;
	}*/
};
} // details

template <
	class T,
	std::size_t N,
	template <class, std::size_t> class S = details::default_storage,
	VectorSettings Settings = VectorSettings{},
	auto OptSettings = SettingsRegistry<SettingsField<"default", SimdSettings{}>>{}>
	requires VecStorage<S, T, N> &&
			 (details::validate_settings<Settings>())
class Vector : public std::conditional_t<Settings.extend_storage, S<T, N>, details::empty_t>
//,
// public std::conditional_t<Settings.arithmetic.enabled, details::arithmetic_overloads, details::empty_t>

{
	[[no_unique_address]]
	std::conditional_t<!Settings.extend_storage, S<T, N>, details::empty_t> m_storage;
public:
	using base_type = T;
	constexpr inline static VectorSettings settings = Settings;
	using Storage = S<T, N>;


	consteval static std::size_t size() noexcept
	{
		return N;
	}

	constexpr decltype(auto) operator[](this Vector const& self, std::size_t i) noexcept
	{
		if constexpr (Settings.extend_storage)
		{
			return self.Storage::operator[](i);
		}
		else {
			return self.m_storage[i];
		}
	}

	constexpr decltype(auto) operator[](this Vector& self, std::size_t i) noexcept
	{
		if constexpr (Settings.extend_storage)
		{
			return self.Storage::operator[](i);
		}
		else {
			return self.m_storage[i];
		}
	}

	friend std::ostream& operator<<(std::ostream& s, const Vector& self)
	{
		static constexpr auto settings = OptSettings.template get<"operator<<(std::ostream& s, const Vector& self)">();
		s << '(';
		details::for_each<N, settings>([&](std::size_t i) {
			if (i != 0)
				s << ", ";
			s << self[i];
		});
		s << ')';

		return s;
	}

	/*
	template <class R, template <class, std::size_t> class _S = S, VectorSettings _Settings = Settings>
	constexpr decltype(auto) map(this const Vector& self, std::move_only_function<R(const T&)>&& fn)
	{
		Vector<R, N, _S, _Settings> ret{};

		details::for_each<N, _Settings.simd>([&](std::size_t i) {
			ret[i] = fn(self[i]);
		});

		return ret;
	}

	template <class R, template <class, std::size_t> class _S = S, VectorSettings _Settings = Settings>
	constexpr decltype(auto) map2(this const Vector& self, const Vector& other, std::move_only_function<R(const T&, const T&)>&& fn)
	{
		Vector<R, N, _S, _Settings> ret{};

		details::for_each<N, _Settings.simd>([&](std::size_t i) {
			ret[i] = fn(self[i], other[i]);
		});

		return ret;
	}

	// {{{ Constructors
	constexpr explicit Vector()
		requires std::default_initializable<Storage>
	: Storage{}
	{
	}

	constexpr Vector(const Vector& other)
		requires std::copyable<T>
	{
		if (this == &other) [[unlikely]]
		{
			throw std::invalid_argument("Attempted to assign to self");
		}

		details::for_each<N, settings.simd>([&, this](std::size_t i) {
			this->operator[](i) = other[i];
		});
	}

	constexpr Vector(Vector&& other)
		requires std::default_initializable<Storage> &&
				 std::is_move_assignable_v<Storage>
	: Storage{ std::exchange(static_cast<Storage&>(other), Storage{}) }
	{
		if (this == &other) [[unlikely]]
		{
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	constexpr Vector(std::initializer_list<T>&& list)
	{
		auto it = list.begin();
		details::for_each<N, settings.simd>([&, this](std::size_t i) {
			this->operator[](i) = *(it++);
		});
	}

	constexpr Vector& operator=(this Vector& self, const Vector& other)
		requires std::copyable<T>
	{
		details::for_each<N, VectorSettings::Simd{
			.enabled = true,
		}>([&](std::size_t i) {
			self[i] = other[i];
		});
		return self;
	}

	constexpr ~Vector()
	{
	}
	*/
	// }}}
}; // Vector

template <class F, std::size_t max_unroll, class... Ts, std::size_t N, template <class, std::size_t> class... Ss, VectorSettings... Ses>
constexpr decltype(auto) map_multi(F&& fn, const Vector<Ts, N, Ss, Ses>&... vecs)
{
	details::for_each<N, max_unroll>([&](std::size_t i) {
		fn(i, vecs[i]...);
	});
}

} // vector



#endif // VECTOR_HPP
