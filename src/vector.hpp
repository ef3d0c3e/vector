#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <array>
#include <concepts>
#include <format>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <iostream>

#include "settings.hpp"

namespace vector {
/// @brief Controls which arithmetic options are enabled
struct Arithmetic
{
	/// Enables arithmetic methods `add`, `mul`, etc.
	bool enabled = true;

	/// Overload operators for arithmetic methods (+, -, etc.)
	///
	/// @note When `implicit_casting` is enabled, the result of
	/// `Vec<T>·Vec<U>` will either be a `Vec<T>` or a `Vec<U>` (where `·` is an
	/// operator). The resulting vector's type is determined by the result of
	/// `T·U`. In cases where `T·U` yields neither a `T` nor a `U`, the operation
	/// will not compile. For instance:
	/// @code
	/// Vec<int, 3> u{1, 2, 3};
	/// Vec<float, 3> v{0.5, 0.1, 0.2};
	/// auto y1 = u + v;
	/// auto y2 = v + u;
	/// assert(v == Vec<float, 3>{1.5, 2.1, 3.2});
	/// assert(v == Vec<float, 3>{1.5, 2.1, 3.2});
	/// @endcode
	bool overloads = true;

	/// Overload assignment operators (+=, /=, etc.)
	bool assignment = true;

	/// Enables operations with scalar (e.g. vec*3.5 or vec+1.0)
	///
	/// @note When `implicit_casting` is enabled, the result of `Vec<T>·U` will be
	/// the result of `T·U` casted to `T` if necessary (where `·` is an operator).
	/// For instance:
	/// @code
	/// Vec<int, 3> v{1, 2, 3};
	/// v *= 1.5;
	/// assert(v == Vec<int, 3>{1, 3, 4});
	/// @endcode
	bool scalar_operations = true;

	/// Allows implicit type casting for arithmetic operations. Generally
	/// `Vector<float> + Vector<int>` is ill formed. Enabling this setting will
	/// allow arithmetic operations for vectors of differnt types.
	///
	/// For instance:
	/// @code
	/// Vec<int, 3> u{1, 2, 3};
	/// Vec<float, 3> v{0.5, 1.4, 0.2};
	/// auto y1 = u.clone().add(v); // requires implicit_casting, because
	/// `int+float` -> float auto y2 = v.clone().add(u); // ok because `float+int`
	/// -> float assert(y1 == Vec<int, 3>{1, 3, 3}); assert(y2 == Vec<float,
	/// 3>{1.5, 3.4, 3.2});
	/// @endcode
	bool implicit_casting = true;
};

/// @brief Controls which formatting options are enabled
struct Formatting
{
	/// Whether to override `operator<<` for ostream
	bool ostream = true;
	/// Specializes `std::formatter` for Vector
	bool format = true;
};

/// @brief Controls which tuple-like options are enabled
struct Tuple
{
	/// Specializes `std::tuple_size` for Vector
	bool size = true;

	/// Specializes `std::tuple_element` for Vector
	bool element = true;

	/// Specializes `std::get` for Vector
	///
	/// @note If the container is stored as a member (i.e `extend_storage` is false), or the
	/// storge does not specializes `std::get`, turning this on enables structured-bindings.
	bool get = true;
};

/// @brief Controls which features are enabled for Vector
struct VectorFeatures
{
	/// Arithmetic features
	Arithmetic arithmetic{};

	/// Formatting features
	Formatting formatting{};

	/// Tuple like features
	Tuple tuple{};

	/// Whether the vector extends it's storage class. Otherwise the storage is
	/// kept as a member
	///
	/// Extending the storage class allows to directly reference the storage when
	/// given a vector:
	/// @code{.cpp}
	/// template <class T>
	/// struct vec3_storage
	/// {
	/// 	T x, y, z;
	/// 	...
	/// };
	/// Vec<float, 3, vec3_storage> v{1.4, 7.8, 0.5};
	/// assert(v.y == 7.8); // extend_storage is on
	/// assert(v[2] == 0.5);
	/// @endcode
	bool extend_storage = true;

	/// Enables iterator
	bool iterator = true;
};

namespace details {
template<class T>
/// @brief Concept for a Vector class
concept is_vector = requires {
	typename T::base_type;
	typename T::Storage;
	{ T::size() } -> std::same_as<std::size_t>;
};

/// @brief Storage for the vector class
///
/// The storage is an aligned memory region that supports the subscript operator
/// `storage[i]`
template<template<class, std::size_t> class S, class T, std::size_t N>
concept vec_storage = requires(S<T, N>&& s, std::size_t i) {
	requires(sizeof(S<T, N>) == sizeof(T) * N);

	// Default constructible
	{ S<T, N>{} } -> std::same_as<S<T, N>>;
	// Moveable
	{ S<T, N>{ std::move(s) } } -> std::same_as<S<T, N>>;
	{ static_cast<S<T, N>&>(s)[i] } -> std::same_as<T&>;
	{ static_cast<const S<T, N>&>(s)[i] } -> std::same_as<const T&>;
};

/// @brief Utility to validate vector features.
/// Meant to be used to give an insight about incompatible settings
///
/// @tparam F The features to validate
template<VectorFeatures F>
consteval bool
validate_features()
{
	// Arithmetic
	static_assert(F.arithmetic.enabled || !F.arithmetic.overloads,
	              "Cannnot enable arithmetic overloads if arithmetic is disabled");
	static_assert(F.arithmetic.enabled || !F.arithmetic.assignment,
	              "Cannnot enable arithmetic assignment if arithmetic "
	              "overloads are disabled");
	static_assert((F.arithmetic.overloads || F.arithmetic.assignment) ||
	                !F.arithmetic.implicit_casting,
	              "Cannnot enable arithmetic implicit casting if operators "
	              "overloading is enabled");

	return true;
}

/// @brief Executes a callback for each elements
///
/// @param fn The callback to execute
/// @tparam N The number of times to execute the callback
/// @tparam simd The callback's execution policy, see vector::SimdSettings
/// @tparam F The callback functor \code{.cpp}(std::size_t) -> void\endcode
template<std::size_t N, SimdSettings simd, class F>
void
for_each(F&& fn)
{
	if constexpr (simd == SimdSettings::UNROLL) {
		[&]<auto... i>(std::index_sequence<i...>) {
			((fn(i)), ...);
		}(std::make_index_sequence<N>{});
	} else if constexpr (simd == SimdSettings::SIMD) {
		[[omp::directive(simd)]]
		for (std::size_t i = 0; i < N; ++i) {
			fn(i);
		}
	} else {
		for (const auto i : std::ranges::iota_view{ 0uz, N }) {
			fn(i);
		}
	}
}

/// @brief Concept for an operator yielding `a·b -> decltype(a)`
template<class Left, class Right, class Op, bool implicit_casting>
concept binary_operator_l = requires(const Left& l, const Right& r, std::size_t i) {
	requires is_vector<Left>;
	requires is_vector<Right>;
	{
		[] {
			using return_t = decltype(Op{}.template operator()(l[i], r[i]));
			if constexpr (implicit_casting) {
				return std::bool_constant <
				           std::is_nothrow_convertible_v<return_t, typename Left::base_type> ||
				         std::same_as < return_t,
				       typename Left::base_type >> {};
			} else {
				return std::bool_constant<std::same_as<return_t, typename Left::base_type>>{};
			}
		}()
	} -> std::same_as<std::true_type>;
};

/// @brief Concept for an operator yielding `a·b -> decltype(b)`
template<class Left, class Right, class Op, bool implicit_casting>
concept binary_operator_r = requires(const Left& l, const Right& r, std::size_t i) {
	requires is_vector<Left>;
	requires is_vector<Right>;
	{
		[] {
			using return_t = decltype(Op{}.template operator()(l[i], r[i]));
			if constexpr (implicit_casting) {
				return std::bool_constant <
				           std::is_nothrow_convertible_v<return_t, typename Right::base_type> ||
				         std::same_as < return_t,
				       typename Right::base_type >> {};
			} else {
				return std::bool_constant<std::same_as<return_t, typename Right::base_type>>{};
			}
		}()
	} -> std::same_as<std::true_type>;
};

/// @brief Concept for an operator yielding `vec·a` -> decltype(vec)
/// @tparam implicit_casting Whether to allow implicit casting, e.g.
/// `vec<int>[..] + float` yields a float, but can be implicitly casted to an int
template<class Vec, class T, class Op, bool implicit_casting>
concept binary_operator_scalar_l = requires(const Vec& vec, const T& scalar, std::size_t i) {
	requires is_vector<Vec>;
	requires(std::is_nothrow_convertible_v<typename Vec::base_type, T>);
	{
		[] {
			using return_t = decltype(Op{}.template operator()(vec[i], scalar));
			if constexpr (implicit_casting) {
				return std::bool_constant <
				           std::is_nothrow_convertible_v<return_t, typename Vec::base_type> ||
				         std::same_as < return_t,
				       typename Vec::base_type >> {};
			} else {
				return std::bool_constant<std::same_as<return_t, typename Vec::base_type>>{};
			}
		}()
	} -> std::same_as<std::true_type>;
};

/// @brief Concept for an assign operator yielding `a·b` -> `&decltype(a)`
template<class Left, class Right, class AssignOp>
concept assign_operator = requires(Left& l, const Right& r, std::size_t i) {
	requires is_vector<Left>;
	requires is_vector<Right>;
	typename Right::base_type;
	{
		AssignOp{}.template operator()(l[i], r[i])
	} -> std::same_as<std::add_lvalue_reference_t<typename Left::base_type>>;
};

/// @brief Concept for an assign operator yielding `vec·a` -> `&decltype(vec)`
template<class Vec, class T, class AssignOp, bool implicit_casting>
concept assign_operator_scalar = requires(Vec& vec, const T& scalar, std::size_t i) {
	requires is_vector<Vec>;
	requires(std::is_nothrow_convertible_v<typename Vec::base_type, T>);
	{
		[] {
			using return_t =
			  std::remove_reference_t<decltype(AssignOp{}.template operator()(vec[i], scalar))>;
			if constexpr (implicit_casting) {
				return std::bool_constant <
				           std::is_nothrow_convertible_v<return_t, typename Vec::base_type> ||
				         std::same_as < return_t,
				       typename Vec::base_type >> {};
			} else {
				return std::bool_constant<std::same_as<return_t, typename Vec::base_type>>{};
			}
		}()
	} -> std::same_as<std::true_type>;
};

/// @brief Defines the arithmetic operations
struct arithmetic
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(is_vector<Self>) && (is_vector<Other>) && (Self::size() == Other::size()) &&    \
	            binary_operator_l<Self,                                                          \
	                              Other,                                                         \
	                              decltype([](auto const& a, auto const& b) -> decltype(auto) {  \
		                              return a __op b;                                           \
	                              }),                                                            \
	                              Self::Features.arithmetic.implicit_casting>                    \
	constexpr Self __op_name(this Self&& self, const Other& other)                               \
	{                                                                                            \
		static constexpr auto settings =                                                         \
		  Self::SimdSettings.template get<#__op_name "(this Self& self, const Other& other)">(); \
		for_each<Self::size(), settings>(                                                        \
		  [&](std::size_t i) { self[i] = self[i] __op other[i]; });                              \
		return std::move(self);                                                                  \
	}                                                                                            \
	template<class Self, class T>                                                                \
	    requires(Self::Features.arithmetic.scalar_operations) && (is_vector<Self>) &&            \
	            (!is_vector<T>) &&                                                               \
	            binary_operator_scalar_l<Self,                                                   \
	                                     T,                                                      \
	                                     decltype([](auto const& a, auto const& b)               \
	                                                -> decltype(auto) { return a __op b; }),     \
	                                     Self::Features.arithmetic.implicit_casting>             \
	constexpr Self __op_name(this Self&& self, const T& scalar)                                  \
	{                                                                                            \
		static constexpr auto settings =                                                         \
		  Self::SimdSettings.template get<#__op_name "(this Self& self, const T& scalar)">();    \
		for_each<Self::size(), settings>([&](std::size_t i) { self[i] = self[i] __op scalar; }); \
		return std::move(self);                                                                  \
	}

	DEFINE_OPERATOR(+, add)
	DEFINE_OPERATOR(-, sub)
	DEFINE_OPERATOR(*, mul)
	DEFINE_OPERATOR(/, div)

#undef DEFINE_OPERATOR
}; // arithmetic

/// @brief Overloads the arithmetic operators (+, -, *, /)
struct arithmetic_overloads
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(is_vector<Self>) && (is_vector<Other>) && (Self::size() == Other::size())       \
	constexpr decltype(auto) operator __op(this const Self& self, const Other& other)            \
	{                                                                                            \
		if constexpr (binary_operator_l<Self,                                                    \
		                                Other,                                                   \
		                                decltype([](const auto& a, const auto& b) {              \
			                                return a __op b;                                     \
		                                }),                                                      \
		                                Self::Features.arithmetic.implicit_casting>) {           \
			static_assert(Self::Features.arithmetic.implicit_casting ||                          \
			              std::is_same_v<typename Self::base_type, typename Other::base_type>);  \
			Self ret{};                                                                          \
			static constexpr auto settings =                                                     \
			  Self::SimdSettings                                                                 \
			    .template get<#__op_name "(this const Self& self, const Other& other)">();       \
			for_each<Self::size(), settings>(                                                    \
			  [&](std::size_t i) { ret[i] = self[i] __op other[i]; });                           \
			return ret;                                                                          \
		} else if constexpr (binary_operator_r<Self,                                             \
		                                       Other,                                            \
		                                       decltype([](const auto& a, const auto& b) {       \
			                                       return a __op b;                              \
		                                       }),                                               \
		                                       Other::Features.arithmetic.implicit_casting>) {   \
			static_assert(Self::Features.arithmetic.implicit_casting ||                          \
			              std::is_same_v<typename Self::base_type, typename Other::base_type>);  \
			Other ret{};                                                                         \
			static constexpr auto settings =                                                     \
			  Other::SimdSettings                                                                \
			    .template get<#__op_name "(this const Self& self, const Other& other)">();       \
			for_each<Self::size(), settings>(                                                    \
			  [&](std::size_t i) { ret[i] = self[i] __op other[i]; });                           \
			return ret;                                                                          \
		} else {                                                                                 \
			[]<bool v = false> { static_assert(v, "Cannot define operator properly"); }();       \
		}                                                                                        \
	}                                                                                            \
	template<class Self, class T>                                                                \
	    requires(Self::Features.arithmetic.scalar_operations) && (is_vector<Self>) &&            \
	            (!is_vector<T>) &&                                                               \
	            binary_operator_scalar_l<Self,                                                   \
	                                     T,                                                      \
	                                     decltype([](auto const& a, auto const& b)               \
	                                                -> decltype(auto) { return a __op b; }),     \
	                                     Self::Features.arithmetic.implicit_casting>             \
	constexpr Self operator __op(this const Self& self, const T& scalar)                         \
	{                                                                                            \
		static constexpr auto settings =                                                         \
		  Self::SimdSettings                                                                     \
		    .template get<#__op_name "(this const Self& self, const T& scalar)">();              \
		Self ret{};                                                                              \
		for_each<Self::size(), settings>([&](std::size_t i) { ret[i] = self[i] __op scalar; });  \
		return ret;                                                                              \
	}

	DEFINE_OPERATOR(+, add)
	DEFINE_OPERATOR(-, sub)
	DEFINE_OPERATOR(*, mul)
	DEFINE_OPERATOR(/, div)

#undef DEFINE_OPERATOR
}; // arithmetic_overloads

/// @brief Overloads the arithmetic assignment operators (+=, -=, *=, /=)
struct arithmetic_assignment_overloads
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(is_vector<Self>) && (is_vector<Other>) && (Self::size() == Other::size()) &&    \
	            assign_operator<Self,                                                            \
	                            Other,                                                           \
	                            decltype([](auto& a, auto const& b) -> decltype(auto) {          \
		                            return a __op b;                                             \
	                            })>                                                              \
	constexpr Self& operator __op(this Self & self, const Other & other)                         \
	{                                                                                            \
		static constexpr auto settings =                                                         \
		  Self::SimdSettings.template get<#__op_name "(this Self& self, const Other& other)">(); \
		for_each<Self::size(), settings>([&](std::size_t i) { self[i] __op other[i]; });         \
		return self;                                                                             \
	}                                                                                            \
	template<class Self, class T>                                                                \
	    requires(Self::Features.arithmetic.scalar_operations) && (is_vector<Self>) &&            \
	            (!is_vector<T>) &&                                                               \
	            assign_operator_scalar<Self,                                                     \
	                                   T,                                                        \
	                                   decltype([](auto& a, auto const& b) -> decltype(auto) {   \
		                                   return a __op b;                                      \
	                                   }),                                                       \
	                                   Self::Features.arithmetic.implicit_casting>               \
	constexpr Self& operator __op(this Self & self, const T & scalar)                            \
	{                                                                                            \
		static constexpr auto settings =                                                         \
		  Self::SimdSettings.template get<#__op_name "(this Self& self, const T& scalar)">();    \
		for_each<Self::size(), settings>([&](std::size_t i) { self[i] __op scalar; });           \
		return self;                                                                             \
	}

	DEFINE_OPERATOR(+=, add_assign)
	DEFINE_OPERATOR(-=, sub_assign)
	DEFINE_OPERATOR(*=, mul_assign)
	DEFINE_OPERATOR(/=, div_assign)

#undef DEFINE_OPERATOR
}; // arithmetic_assignment_overloads
} // namespace details

/// @brief The vector class
///
/// @tparam T The vector's element type
/// @tparam N The number of elements
/// @tparam S The storage type, must verify details::vec_storage. Defaults to `std::array`
/// @tparam _Features The vector's features, see VectorFeatures
/// @tparam _SimdSettings Dispatch policies for details::for_each, see
/// vector::SettingsRegistry< SettingsField< Names, Settings >... >
template<class T,
         std::size_t N,
         template<class, std::size_t> class S = std::array,
         VectorFeatures _Features = VectorFeatures{},
         auto _SimdSettings = SettingsRegistry<SettingsField<"default", SimdSettings{}>>{}>
    requires details::vec_storage<S, T, N> && (details::validate_features<_Features>())
struct Vector
  : public std::conditional_t<_Features.extend_storage, S<T, N>, std::monostate>
  , public std::conditional_t<_Features.arithmetic.enabled, details::arithmetic, std::monostate>
  , public std::
      conditional_t<_Features.arithmetic.overloads, details::arithmetic_overloads, std::monostate>
  , public std::conditional_t<_Features.arithmetic.assignment,
                              details::arithmetic_assignment_overloads,
                              std::monostate>
{
	[[no_unique_address]]
	std::conditional_t<!_Features.extend_storage, S<T, N>, std::monostate> _storage;

	/// The vector's SimdSettings i.e. template parameter `_SimdSettings`
	constexpr inline static auto SimdSettings = _SimdSettings;
	/// The vector's VectorFeatures i.e. template parameter `_Features`
	constexpr inline static VectorFeatures Features = _Features;
	/// The vector's base type i.e. template parameter `T`
	using base_type = T;
	/// The vector's storage type i.e. template parameter `S`
	using Storage = S<T, N>;

	/// @brief Gets the vector's size i.e. template parameter `N`
	///
	/// @return The number of elements
	consteval static std::size_t size() noexcept { return N; }

	/// @brief Subscript operator
	///
	/// @param self A (const) lvalue reference to Vetor
	/// @param i Index of the element
	///
	/// @returns The value at index @p i in @p self
	constexpr decltype(auto) operator[](this auto&& self, std::size_t i) noexcept
	{
		if constexpr (Features.extend_storage) {
			return self.Storage::operator[](i);
		} else {
			return self._storage[i];
		}
	}

	friend std::ostream& operator<<(std::ostream& s, const Vector& self)
	    requires(Features.formatting.ostream)
	{
		static constexpr auto settings =
		  SimdSettings.template get<"operator<<(std::ostream& s, const Vector& self)">();
		s << '(';
		details::for_each<N, settings>([&](std::size_t i) {
			if (i != 0)
				s << ", ";
			s << self[i];
		});
		s << ')';

		return s;
	}

	// {{{ Comparisons
	constexpr bool operator==(this const Vector& self, const Vector& other)
	{
		static constexpr auto settings =
		  SimdSettings.template get<"operator==(this const Vector& self, const Vector& other)">();
		bool equal = true;
		details::for_each<size(), settings>([&](std::size_t i) { equal &= self[i] == other[i]; });
		return equal;
	}

	constexpr bool operator!=(this const Vector& self, const Vector& other)
	{
		return !(self == other);
	}
	// }}}

	// {{{ Constructors
	/// @brief Default constructor
	///
	/// Construct by calling vector::vec_storage's default constructor
	constexpr explicit Vector()
	    requires(Features.extend_storage) && std::default_initializable<Storage>
	  : Storage{}
	{
	}

	/// @brief Default constructor
	///
	/// Construct by calling vector::vec_storage's default constructor
	constexpr explicit Vector()
	    requires(!Features.extend_storage) && std::default_initializable<Storage>
	  : _storage{}
	{
	}

	/// @brief Copy constructor for non-extended storage that implements copy
	///
	/// This is called if the storage is not extended (i.e. `extend_storage` is disabled) and the
	/// storage has a copy constructor
	constexpr Vector(const Vector& other)
	    requires(!Features.extend_storage) && std::copyable<Storage>
	  : _storage{ other._storage }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Copy constructor for extended storage that implements copy
	///
	/// This is called if the storage is extended (i.e. `extend_storage` is enabled) and the
	/// storage has a copy constructor
	constexpr Vector(const Vector& other)
	    requires(Features.extend_storage) && std::copyable<Storage>
	  : Storage{ other }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Copy constructor for non-copyable storage that have copyable element
	constexpr Vector(const Vector& other)
	    requires std::copyable<T> && (!std::copyable<Storage>)
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}

		static constexpr auto settings =
		  SimdSettings.template get<"Vector(const Vector& other)">();
		details::for_each<N, settings>(
		  [&, this](std::size_t i) { this->operator[](i) = other[i]; });
	}

	constexpr Vector(Vector&& other)
	    requires(Features.extend_storage) && std::is_move_assignable_v<Storage>

	  : Storage{ std::move(static_cast<Storage&>(other)) }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	constexpr Vector(Vector&& other)
	    requires(!Features.extend_storage) && std::is_move_assignable_v<Storage>

	  : _storage{ std::move(other._storage) }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	constexpr Vector(std::initializer_list<T>&& list)
	{
		if (list.size() != size()) [[unlikely]] {
			throw std::invalid_argument("Invalid initializer_list size");
		}

		auto it = list.begin();
		static constexpr auto settings =
		  SimdSettings.template get<"Vector(std::initializer_list<T>&& list)">();
		details::for_each<N, settings>(
		  [&, this](std::size_t i) { this->operator[](i) = *(it++); });
	}

	constexpr Vector& operator=(this Vector& self, const Vector& other)
	    requires std::copyable<T>
	{
		if (&self == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}

		static constexpr auto settings =
		  SimdSettings.template get<"operator=(this Vector& self, const Vector& other)">();
		details::for_each<N, settings>([&](std::size_t i) { self[i] = other[i]; });
		return self;
	}

	/// @brief Move assignment
	constexpr Vector& operator=(this Vector& self, Vector&& other)
	    requires std::is_move_assignable_v<Storage>
	{
		if (&self == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}

		if constexpr (Features.extend_storage) {
			self = std::move(other);
		} else {
			self._storage = std::move(other._storage);
		}

		return self;
	}

	/// @brief Clones the vector
	///
	/// Clones the vector by calling the copy construcument 1 of ‘constexpr Self&
	/// vector::details::arithmetic::add(this Self&, const Other&) [wtor
	///
	/// @pram self Vector to clone
	constexpr Vector clone(this const Vector& self) { return Vector{ self }; }

	constexpr ~Vector() {}
	// }}}
}; // Vector
} // namespace vector

/// Specialization of std's functionnalities for Vector
namespace std {
/// @brief `std::formatter` specialization for Vector
template<class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::details::is_vector<::vector::Vector<T, N, S, Features, SimdSettings>> &&
             (Features.formatting.format)
struct formatter<::vector::Vector<T, N, S, Features, SimdSettings>, char>
{

	std::formatter<T, char> element_formatter;

	template<class ParseContext>
	constexpr ParseContext::iterator parse(ParseContext& ctx)
	{
		return element_formatter.template parse(ctx);
	}

	template<class FmtContext>
	FmtContext::iterator format(auto&& s, FmtContext& ctx) const
	{
		auto out = ctx.out();
		*out++ = '(';
		for (const auto i : std::ranges::iota_view{ 0uz, s.size() }) {
			if (i != 0) {
				*out++ = ',';
				*out++ = ' ';
			}
			out = element_formatter.format(s[i], ctx);
		}
		*out++ = ')';

		return out;
	}
};

// @brief `std::tuple_size` specialization for Vector
template<class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::details::is_vector<::vector::Vector<T, N, S, Features, SimdSettings>> &&
             (Features.tuple.size)
struct tuple_size<::vector::Vector<T, N, S, Features, SimdSettings>>
  : std::integral_constant<std::size_t, N>
{};

// @brief `std::tuple_element` specialization for Vector
template<std::size_t I,
         class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::details::is_vector<::vector::Vector<T, N, S, Features, SimdSettings>> &&
             (Features.tuple.element)
struct tuple_element<I, ::vector::Vector<T, N, S, Features, SimdSettings>>
{
	using type = T;
};

// @brief `std::get` specialization for Vector
template<std::size_t I,
         class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::details::is_vector<::vector::Vector<T, N, S, Features, SimdSettings>> &&
             (Features.tuple.get)
constexpr decltype(auto)
  get(const ::vector::Vector<T, N, S, Features, SimdSettings>& vec) noexcept
{
	return vec[I];
}

// @brief `std::get` specialization for Vector
template<std::size_t I,
         class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::details::is_vector<::vector::Vector<T, N, S, Features, SimdSettings>> &&
             (Features.tuple.get)
constexpr decltype(auto) get(::vector::Vector<T, N, S, Features, SimdSettings>& vec) noexcept
{
	return vec[I];
}
} // namespace std

#endif // VECTOR_HPP
