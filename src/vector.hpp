#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <array>
#include <concepts>
#include <format>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include <iostream>

namespace vector {
namespace details {
/// @brief Literal string as NNTP
///
/// @tparam N length of string
template<std::size_t N>
struct literal
{
	char data[N];

	constexpr literal(const char (&literal)[N]) { std::copy_n(literal, N, data); }

	template<std::size_t M>
	constexpr bool operator==(this const literal<N>& self, const literal<M>& other)
	{
		if constexpr (M != N)
			return false;

		for (const auto i : std::ranges::iota_view{ 0uz, N }) {
			if (self.data[i] != other.data[i])
				return false;
		}

		return true;
	}
};
} // namespace details

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
	/// storage does not specializes `std::get` (or is not structural), turning this on enables
	/// structured-bindings. For instance:
	/// @code
	/// template <class T, std::size_t N>
	/// struct Storage {
	/// 	std::vector<T> data; // Not structural, no std::get specialization
	///		...
	/// };
	/// Vec<int, 3, Storage> v{5, 7, 9};
	/// const auto& [x, y, z] = v; // Features.tuple.get enabled
	/// @endcode
	bool get = true;
};

/// @brief Controls the iterators settings
struct Iterators
{
	/// Enables iterators on vector::Vector.
	/// Because `vec_storage` is contiguous, the iterators are pointers to `T`.
	bool enabled = true;

	/// Uses the storage's iterator when available
	bool use_storage = false;
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

	/// Iterators settings
	Iterators iterators{};

	/// Whether the vector extends it's storage class. Otherwise the storage is
	/// kept as a member.
	///
	/// Extending the storage class allows to directly reference the storage when
	/// given a vector:
	/// @code{.cpp}
	/// template <class T, size_t N>
	/// 	requires (N == 3)
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

	/// Enables the implicit copy constructor.
	///
	/// When disabled, `clone()` has to be called explicitly
	/// @code
	/// Vec<int, 3> u;
	/// auto v = u; // requires copy_constructor
	/// auto v = u.clone(); // ok
	/// Vec<int, 3> v;
	/// v = u; // ok, element-wise copy
	/// @endcode
	bool copy_constructor = false;
};

// Settings for for_each function
enum SimdSettings
{
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
template<details::literal _key, SimdSettings _value>
struct SettingsField
{
	constexpr static inline details::literal key = _key;
	constexpr static inline SimdSettings value = _value;
};

/// @cond
template<class...>
class SettingsRegistry;
/// @endcond

/**
 * @brief Registry holding vector::SimdSettings for the vector's dispatch policy
 *
 * @tparam SettingsField A (key, value) pair for the setting's name and value
 * \code{.cpp}
 * SettingsField<"default", SimdSettings::SIMD>, // required
 * SettingsField<"add(this Self& self, const Other& other)", SimdSettings::UNROLL>, // unroll for
 * add
 * ...
 * \endcode
 */
template<details::literal... Names, SimdSettings... Settings>
class SettingsRegistry<SettingsField<Names, Settings>...>
{
	using settings = std::tuple<SettingsField<Names, Settings>...>;

	template<std::size_t I, details::literal key>
	static consteval decltype(auto) get_impl()
	{
		using elem = std::tuple_element_t<I, settings>;
		if constexpr (elem::key == key) {
			return (elem::value);
		} else if constexpr (I + 1 < std::tuple_size_v<settings>) {
			return get_impl<I + 1, key>();
		} else {
			if constexpr (key == details::literal<8>{ "default" }) {
				// No default key
				[]<bool v = true>() { static_assert(v, "No default key for settings"); }();
			} else {
				// Get default key if not found for current key
				return get_impl<0, "default">();
			}
		}
	}

	public:
	template<details::literal key>
	static consteval decltype(auto) get()
	{
		return get_impl<0, key>();
	}
};

/// @brief Concept for a Vector class
template<class T>
concept vector_type = requires {
	typename T::base_type;
	typename T::Storage;
	{ T::size() } -> std::same_as<std::size_t>;
};

namespace details {
/// @brief Checks if a type implements tuple_element
template<class T, std::size_t I>
concept has_tuple_element = requires(T t) {
	typename std::tuple_element_t<I, std::remove_const_t<T>>;
	{ std::get<I>(t) } -> std::convertible_to<const std::tuple_element_t<I, T>&>;
};

/// @brief Checks if a type is tuple-like
template<class T>
concept tuple_like = requires(T t) {
	requires !std::is_reference_v<T>;
	typename std::tuple_size<T>::type;
	requires std::derived_from<std::tuple_size<T>,
	                           std::integral_constant<std::size_t, std::tuple_size_v<T>>>;
} && []<std::size_t... N>(std::index_sequence<N...>) {
	return (has_tuple_element<T, N> && ...);
}(std::make_index_sequence<std::tuple_size_v<T>>());

/// @brief Storage for the vector class
///
/// The storage is an aligned memory region that supports the subscript operator
/// `storage[i]`. In addition, the storage must be default-constructible and move-constructible
///
template<template<class, std::size_t> class S, class T, std::size_t N>
concept vec_storage = requires(S<T, N>&& s, std::size_t i) {
	requires(sizeof(S<T, N>) == sizeof(T) * N);

	// Default constructible
	{ S<T, N>{} } -> std::same_as<S<T, N>>;
	// Moveable
	{ S<T, N>{ std::move(s) } } -> std::same_as<S<T, N>>;
	// Subscript operator
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

	static_assert(F.iterators.enabled || !F.iterators.use_storage,
	              "Can't use storage iterators if iterators are disabled");

	return true;
}

/// @brief Executes a callback for each elements
///
/// @param fn The callback to execute
/// @tparam N The number of times to execute the callback
/// @tparam simd The callback's execution policy, see vector::SimdSettings
/// @tparam F The callback functor @code{.cpp}(std::size_t) -> void@endcode
template<std::size_t N, SimdSettings simd, class F>
void
for_each(F&& fn)
{
	if constexpr (simd == SimdSettings::UNROLL) {
		[&]<auto... i>(std::index_sequence<i...>) {
			((fn(i)), ...);
		}(std::make_index_sequence<N>{});
	} else if constexpr (simd == SimdSettings::SIMD) {
		//[[omp::directive(simd)]]
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
	requires vector_type<Left>;
	requires vector_type<Right>;
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
	requires vector_type<Left>;
	requires vector_type<Right>;
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
	requires vector_type<Vec>;
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
	requires vector_type<Left>;
	requires vector_type<Right>;
	typename Right::base_type;
	{
		AssignOp{}.template operator()(l[i], r[i])
	} -> std::same_as<std::add_lvalue_reference_t<typename Left::base_type>>;
};

/// @brief Concept for an assign operator yielding `vec·a` -> `&decltype(vec)`
template<class Vec, class T, class AssignOp, bool implicit_casting>
concept assign_operator_scalar = requires(Vec& vec, const T& scalar, std::size_t i) {
	requires vector_type<Vec>;
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

/// @cond
template<bool>
struct arithmetic;
template<>
struct arithmetic<false>
{};
/// @endcond

/// @brief Defines the arithmetic operations
template<>
struct arithmetic<true>
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(vector_type<Self>) && (vector_type<Other>) &&                                   \
	            (Self::size() == Other::size()) &&                                               \
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
	    requires(Self::Features.arithmetic.scalar_operations) && (vector_type<Self>) &&          \
	            (!vector_type<T>) &&                                                             \
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

/// @cond
template<bool>
struct arithmetic_overloads;
template<>
struct arithmetic_overloads<false>
{};
/// @endcond

/// @brief Overloads the arithmetic operators (+, -, *, /)
template<>
struct arithmetic_overloads<true>
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(vector_type<Self>) && (vector_type<Other>) && (Self::size() == Other::size())   \
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
	    requires(Self::Features.arithmetic.scalar_operations) && (vector_type<Self>) &&          \
	            (!vector_type<T>) &&                                                             \
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

/// @cond
template<bool>
struct arithmetic_assignment_overloads;
template<>
struct arithmetic_assignment_overloads<false>
{};
/// @endcond

/// @brief Overloads the arithmetic assignment operators (+=, -=, *=, /=)
template<>
struct arithmetic_assignment_overloads<true>
{
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
	template<class Self, class Other = Self>                                                     \
	    requires(vector_type<Self>) && (vector_type<Other>) &&                                   \
	            (Self::size() == Other::size()) &&                                               \
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
	    requires(Self::Features.arithmetic.scalar_operations) && (vector_type<Self>) &&          \
	            (!vector_type<T>) &&                                                             \
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

/// @cond
template<bool>
struct iterators;

template<>
struct iterators<false>
{};
/// @endcond

/// @brief Iterators helper for custom iterators
template<>
struct iterators<true>
{
	/// @brief Iterator type
	///
	/// @tparam Const whether this is a `const_iterator`
	/// @tparam Reverse whether this is a `reverse_iterator`
	template<class Self, bool Const, bool Reverse>
	struct vector_iterator
	{
		using iterator_category = std::
		  conditional_t<Reverse, std::random_access_iterator_tag, std::contiguous_iterator_tag>;
		using value_type = Self::base_type;
		using difference_type = std::ptrdiff_t;
		using pointer = std::conditional_t<Const, const value_type*, value_type*>;
		using reference = std::conditional_t<Const, const value_type&, value_type&>;

		constexpr std::ptrdiff_t offset(this const vector_iterator& self, std::ptrdiff_t off)
		{
			if constexpr (Reverse) {
				return std::ptrdiff_t{ self.pos } - off;
			} else {
				return self.pos + off;
			}
		}

		std::conditional_t<Const, const Self*, Self*> vec;
		std::ptrdiff_t pos;

		constexpr decltype(auto) operator*(this auto&& self) { return (*self.vec)[self.pos]; }
		constexpr decltype(auto) operator->(this auto&& self) { return &(*self.vec)[self.pos]; }
		constexpr decltype(auto) operator[](this auto&& self, std::size_t i)
		{
			return (*self.vec)[self.offset(i)];
		}

		constexpr vector_iterator& operator=(this vector_iterator& self,
		                                     const vector_iterator& other) noexcept
		{
			self.vec = other.vec;
			self.pos = other.pos;

			return self;
		}

		constexpr vector_iterator& operator++(this vector_iterator& self)
		{
			self.pos = self.offset(1);
			return self;
		}
		constexpr vector_iterator operator++(this const vector_iterator& self, int)
		{
			auto copy = self;
			return ++copy;
		}
		constexpr vector_iterator& operator--(this vector_iterator& self)
		{
			self.pos = self.offset(-1);
			return self;
		}
		constexpr vector_iterator operator--(this const vector_iterator& self, int)
		{
			auto copy = self;
			return --copy;
		}

		constexpr vector_iterator operator+(this const vector_iterator& self,
		                                    const difference_type i)
		{
			return vector_iterator{ .vec = self.vec, .pos = self.offset(i) };
		}
		friend constexpr vector_iterator operator+(const difference_type i,
		                                           const vector_iterator& self)
		{
			return vector_iterator{ .vec = self.vec, .pos = self.offset(i) };
		}
		constexpr vector_iterator& operator+=(this vector_iterator& self, difference_type i)
		{
			self.pos += i;
			return self;
		}
		constexpr vector_iterator operator-(this const vector_iterator& self, difference_type i)
		{
			return vector_iterator{ .vec = self.vec, .pos = self.offset(-i) };
		}
		constexpr difference_type operator-(this const vector_iterator& self,
		                                    const vector_iterator& other)
		{
			return self.vec + self.offset(-other.pos);
		}
		constexpr vector_iterator& operator-=(this vector_iterator& self, difference_type i)
		{
			self.pos = self.offset(-i);
			return self;
		}

		constexpr bool operator==(this const vector_iterator& self, const vector_iterator& other)
		{
			return self.vec == other.vec && self.pos == other.pos;
		}
		constexpr bool operator!=(this const vector_iterator& self, const vector_iterator& other)
		{
			return !(self == other);
		}

		constexpr auto operator<=>(this const vector_iterator& self, const vector_iterator& other)
		{
			if (self.vec != other.vec)
				return std::strong_ordering::equivalent;
			return self.pos <=> other.pos;
		}
	};

	/// @brief Iterator to the Vector's start
	template<class Self>
	constexpr decltype(auto) begin(this Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::begin(self._storage);
		} else {
			return vector_iterator<Self, false, false>{
				.vec = &self,
				.pos = 0,
			};
		}
	}
	/// @brief Iterator to the Vector's end
	template<class Self>
	constexpr decltype(auto) end(this Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::end(self._storage);
		} else {
			return vector_iterator<Self, false, false>{
				.vec = &self,
				.pos = Self::size(),
			};
		}
	}
	/// @brief Const iterator to the Vector's begin
	template<class Self>
	constexpr decltype(auto) cbegin(this const Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::cbegin(self._storage);
		} else {
			return vector_iterator<Self, true, false>{
				.vec = &self,
				.pos = 0,
			};
		}
	}
	/// @brief Const iterator to the Vector's end
	template<class Self>
	constexpr decltype(auto) cend(this const Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::cend(self._storage);
		} else {
			return vector_iterator<Self, true, false>{
				.vec = &self,
				.pos = Self::size(),
			};
		}
	}
	/// @brief Reverse iterator to the Vector's start
	template<class Self>
	constexpr decltype(auto) rbegin(this Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::rbegin(self._storage);
		} else {
			return vector_iterator<Self, false, true>{
				.vec = &self,
				.pos = Self::size() - 1,
			};
		}
	}

	/// @brief Reverse iterator to the Vector's end
	template<class Self>
	constexpr decltype(auto) rend(this Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::rend(self._storage);
		} else {
			return vector_iterator<Self, false, true>{
				.vec = &self,
				.pos = -1,
			};
		}
	}
	/// @brief Const reverse iterator to the Vector's start
	template<class Self>
	constexpr decltype(auto) crbegin(this const Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage) {
			return std::crbegin(self._storage);
		} else {
			return vector_iterator<Self, true, true>{
				.vec = &self,
				.pos = Self::size() - 1,
			};
		}
	}
	/// @brief Const reverse iterator to the Vector's end
	template<class Self>
	constexpr decltype(auto) crend(this const Self& self)
	{
		if constexpr (Self::Features.iterators.use_storage && !Self::Features.extend_storage) {
			return std::crend(self._storage);
		} else {
			return vector_iterator<Self, true, true>{
				.vec = &self,
				.pos = -1,
			};
		}
	}
};
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
  : std::conditional_t<_Features.extend_storage, S<T, N>, std::monostate>
  , details::arithmetic<_Features.arithmetic.enabled>
  , details::arithmetic_overloads<_Features.arithmetic.overloads>
  , details::arithmetic_assignment_overloads<_Features.arithmetic.assignment>
  , details::iterators<_Features.iterators.enabled && !_Features.extend_storage>
{
	/// Stored data if `extend_storage` is disabled
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
	/// The iterator type
	using iterator = decltype([] {
		if constexpr (!Features.iterators.enabled) {
			return std::monostate{};
		} else if constexpr (Features.iterators.use_storage) {
			return std::begin(Storage{});
		} else {
			return details::iterators<
			  true>::vector_iterator<Vector<T, N, S, Features, SimdSettings>, false, false>{};
		}
	}());
	using const_iterator = decltype([] {
		if constexpr (!Features.iterators.enabled) {
			return std::monostate{};
		} else if constexpr (Features.iterators.use_storage) {
			return std::cbegin(Storage{});
		} else {
			return details::iterators<
			  true>::vector_iterator<Vector<T, N, S, Features, SimdSettings>, true, false>{};
		}
	}());
	using reverse_iterator = decltype([] {
		if constexpr (!Features.iterators.enabled) {
			return std::monostate{};
		} else if constexpr (Features.iterators.use_storage) {
			return std::rbegin(Storage{});
		} else {
			return details::iterators<
			  true>::vector_iterator<Vector<T, N, S, Features, SimdSettings>, false, true>{};
		}
	}());
	using const_reverse_iterator = decltype([] {
		if constexpr (!Features.iterators.enabled) {
			return std::monostate{};
		} else if constexpr (Features.iterators.use_storage) {
			return std::crbegin(Storage{});
		} else {
			return details::iterators<
			  true>::vector_iterator<Vector<T, N, S, Features, SimdSettings>, true, true>{};
		}
	}());

	/// @brief Gets the vector's size i.e. template parameter `N`
	///
	/// @return The number of elements
	consteval static std::size_t size() noexcept { return N; }

	/// @brief Subscript operator
	///
	/// @param self A (const) lvalue reference to Vector
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

	/// @brief Copy constructor (for extended storage)
	/// @param other Vector to copy
	///
	/// This is called if the storage is extended (i.e. `extend_storage` is enabled) and the
	/// storage is copyable
	constexpr Vector(const Vector& other)
	    requires(Features.copy_constructor) && (Features.extend_storage) && std::copyable<Storage>
	  : Storage{ other }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Copy constructor (for non extended storage)
	/// @param other Vector to copy
	///
	/// This is called if the storage is not extended (i.e. `extend_storage` is disabled) and the
	/// storage is copyable.
	constexpr Vector(const Vector& other)
	    requires(Features.copy_constructor) &&
	            (!Features.extend_storage) && std::copyable<Storage>
	  : _storage{ other._storage }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Copy constructor (element-wise copy)
	/// @param other Vector to copy
	///
	/// This is called if the storage is not copyable but the stored elements are copyable.
	///
	/// @note This methods requires the default constructor, so it may be more expensive than
	/// other copy constructors.
	constexpr Vector(const Vector& other)
	    requires(Features.copy_constructor) && (!std::copyable<Storage>) && std::copyable<T>
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}

		*this = other;
	}

	/// @brief Move constructor (for extended storage)
	/// @param other Vector to move int self
	constexpr Vector(Vector&& other)
	    requires(Features.extend_storage) && std::movable<Storage>

	  : Storage{ std::move(static_cast<Storage&&>(other)) }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Move constructor (for non extended storage)
	/// @param other Vector to move int self
	constexpr Vector(Vector&& other)
	    requires(!Features.extend_storage) && std::movable<Storage>

	  : _storage{ std::move(other._storage) }
	{
		if (this == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}
	}

	/// @brief Storage constructor (for non extended storage)
	/// @param storage Storage to move into self
	constexpr Vector(Storage&& storage)
	    requires(!Features.extend_storage) && std::movable<Storage>
	  : _storage{ std::move(storage) }

	{
	}

	/// @brief Storage constructor (for extended storage)
	/// @param storage Storage to move into self
	constexpr Vector(Storage&& storage)
	    requires(Features.extend_storage) && std::movable<Storage>
	  : Storage{ std::move(storage) }

	{
	}

	/// @brief Initializer list constructor
	///
	/// @note Will throw if the initializer_list's size if different from Vector::size()
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

	/// @brief Copy assignment
	/// @param other Vector to copy element-wise into self
	constexpr Vector& operator=(this Vector& self, const Vector& other)
	    requires std::copyable<T>
	{
		std::cout << "= called" << std::endl;
		if (&self == &other) [[unlikely]] {
			throw std::invalid_argument("Attempted to assign to self");
		}

		static constexpr auto settings =
		  SimdSettings.template get<"operator=(this Vector& self, const Vector& other)">();
		details::for_each<N, settings>([&](std::size_t i) { self[i] = other[i]; });
		return self;
	}

	/// @brief Move assignment
	/// @param other Vector to move into self
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
	/// Clones the vector by either cloning it's storage,
	/// or by default-initializing the vector and copying element-wise.
	///
	/// @pram self Vector to clone
	constexpr Vector clone(this const Vector& self)
	{
		// Copy storage
		if constexpr (!Features.extend_storage && std::copyable<Storage>) {
			return Vector{ Storage{ self._storage } };
		} else if constexpr (Features.extend_storage && std::copyable<Storage>) {
			auto copy = static_cast<Storage>(self);
			return static_cast<Vector>(std::move(copy));
		}
		// Copy element-wise (more expensive since it requires the default constructor)
		else if constexpr (std::copyable<T>) {
			auto ret = Vector{};

			ret = self;
			return ret;
		} else {
			[]<bool v = false> {
				static_assert(v,
				              "Cannot clone(), T is not copyable or the Storage is not copyable");
			}();
		}
	}

	/// @brief Destructor
	constexpr ~Vector() {}
	// }}}
}; // Vector
} // namespace vector

/// Specialization of std's functionalities for vector::Vector
namespace std {
/// @brief `std::formatter` specialization for link vector::Vector
/// @related vector::Vector
///
/// @tparam T Formatted element.
/// The format string is parsed using \p T's `std::formatter`.
template<class T,
         std::size_t N,
         template<class, std::size_t>
         class S,
         ::vector::VectorFeatures Features,
         auto SimdSettings>
    requires ::vector::vector_type<::vector::Vector<T, N, S, Features, SimdSettings>> &&
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

/// @brief `std::tuple_size` specialization for vector::Vector
/// @related vector::Vector
template<::vector::vector_type Vec>
    requires(Vec::Features.tuple.size)
struct tuple_size<Vec> : std::integral_constant<std::size_t, Vec::size()>
{};

/// @brief `std::tuple_element` specialization for vector::Vector
/// @related vector::Vector
template<std::size_t I, ::vector::vector_type Vec>
    requires(Vec::Features.tuple.element)
struct tuple_element<I, Vec>
{
	using type = Vec::base_type;
};
/// @brief `std::get` specialization for vector::Vector
/// @related vector::Vector
template<std::size_t I, ::vector::vector_type Vec>
    requires(Vec::Features.tuple.get) && (::vector::details::tuple_like<typename Vec::Storage>)
constexpr decltype(auto) get(const Vec& vec)
{
	return vec[I];
}

/// @brief `std::get` specialization for vector::Vector
/// @related vector::Vector
template<std::size_t I, ::vector::vector_type Vec>
    requires(Vec::Features.tuple.get) && (::vector::details::tuple_like<typename Vec::Storage>)
constexpr decltype(auto) get(Vec& vec)
{
	return vec[I];
}
} // namespace std

/// @brief `get` specialization for vector::Vector
/// @related vector::Vector
///
/// @note This function is called when the storage is not structural, otherwise std::get is called
template<std::size_t I, ::vector::vector_type Vec>
    requires(Vec::Features.tuple.get) && (!vector::details::tuple_like<typename Vec::Storage>)
constexpr decltype(auto) get(const Vec& vec)
{
	return vec[I];
}
/// @brief `get` specialization for vector::Vector
/// @related vector::Vector
///
/// @note This function is called when the storage is not structural, otherwise std::get is called
template<std::size_t I, ::vector::vector_type Vec>
    requires(Vec::Features.tuple.get) && (!vector::details::tuple_like<typename Vec::Storage>)
constexpr decltype(auto) get(Vec& vec)
{
	return vec[I];
}

#endif // VECTOR_HPP
