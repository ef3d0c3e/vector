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

namespace vector {
/// @brief Storage for the vector class
///
/// The storage is an aligned memory region that supports the subscript operator `storage[i]`
template <template <class, std::size_t> class S, class T, std::size_t N>
concept vec_storage = requires(S<T, N> &&s, std::size_t i) {
    requires(sizeof(S<T, N>) == sizeof(T) * N);

    { static_cast<S<T, N> &>(s)[i] } -> std::same_as<T &>;
    { static_cast<const S<T, N> &>(s)[i] } -> std::same_as<const T &>;
};

/// Settings for the Vector class
struct VectorSettings {
    /// Settings the the vector's arithmetic
    struct Arithmetic {
        /// Enables arithmetic methods `add`, `mul`, etc.
        bool enabled = true;

        /// Enables operations with scalar (e.g. vec*3.5 or vec+1.0)
        bool scalar_operations = true;

        /// Overload operators for arithmetic methods (+, -, etc.)
        bool overloads = true;

        /// Overload assignment operators (+=, /=, etc.)
        bool assignment = true;

        /// Allows implicit type casting for operators
        /// Generally `Vector<float> + Vector<int>` is ill formed
        /// Enabling this settings will cast the result type to the type of the
        /// lhs
        bool implicit_casting = false;
    } arithmetic{};

    /// Whether the vector extends it's storage class
    /// Otherwise the storage is kept as a member
    bool extend_storage = true;
};

namespace details {
/// Utility to validate vector settings.
/// Meant to be used to give an insight about incompatible settings
///
/// @tparam S The settings to validate
template <VectorSettings S> consteval bool validate_settings() {
    // Arithmetic
    static_assert(S.arithmetic.enabled || !S.arithmetic.overloads,
                  "Cannnot enable arithmetic overloads if arithmetic is disabled");
    static_assert(S.arithmetic.enabled || !S.arithmetic.assignment,
                  "Cannnot enable arithmetic assignment if arithmetic "
                  "overloads are disabled");
    static_assert((S.arithmetic.overloads || S.arithmetic.assignment) ||
                      !S.arithmetic.implicit_casting,
                  "Cannnot enable arithmetic implicit casting if operators "
                  "overloading is enabled");

    return true;
}

/// Default vector storage, an aligned `T[N]` array
template <class T, std::size_t N> struct default_storage {
    T data[N];

    constexpr inline decltype(auto) operator[](this auto &self, std::size_t i) noexcept {
        return (self.data[i]);
    }
};
static_assert(vec_storage<default_storage, float, 5>);

template <std::size_t N, SimdSettings simd, class F> void for_each(F &&fn) {
    if constexpr (simd == SimdSettings::UNROLL) {
        {
            [&]<auto... i>(std::index_sequence<i...>) {
                ((fn(i)), ...);
            }(std::make_index_sequence<N>{});
        }
    } else if constexpr (simd == SimdSettings::SIMD) {
        [[omp::directive(simd)]]
        for (std::size_t i = 0; i < N; ++i) {
            fn(i);
        }
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            fn(i);
        }
    }
}

template <class Left, class Right, class Op>
concept binary_operator_l = requires(const Left &l, const Right &r, std::size_t i) {
    typename Left::base_type;
    typename Right::base_type;
    { Op{}.template operator()(l[i], r[i]) } -> std::same_as<typename Left::base_type>;
};

template <class Left, class Right, class Op>
concept binary_operator_r = requires(const Left &l, const Right &r, std::size_t i) {
    typename Left::base_type;
    typename Right::base_type;
    { Op{}.template operator()(l[i], r[i]) } -> std::same_as<typename Right::base_type>;
};

template <class Vec, class T, class Op, bool implicit_casting>
concept binary_operator_scalar_l = requires(const Vec &vec, const T &scalar, std::size_t i) {
    typename Vec::base_type;
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

template <class Left, class Right, class AssignOp>
concept assign_operator = requires(Left &l, const Right &r, std::size_t i) {
    typename Left::base_type;
    typename Right::base_type;
    { AssignOp{}.template operator()(l[i], r[i]) } -> std::same_as<std::add_lvalue_reference_t<typename Left::base_type>>;
};

struct empty_t {};

/// Defines the arithmetic operations
struct arithmetic {
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
    template <class Self, class Other = Self>                                                    \
        requires binary_operator_l<Self, Other,                                                  \
                                   decltype([](auto const &a, auto const &b) -> decltype(auto) { \
                                       return a __op b;                                          \
                                   })>                                                           \
    constexpr Self __op_name(this Self &self, const Other &other) {                              \
        static constexpr auto settings =                                                         \
            Self::SimdSettings                                                                   \
                .template get<#__op_name "(this const Self& self, const Other& other)">();       \
        for_each<Self::size(), settings>(                                                        \
            [&](std::size_t i) { self[i] = self[i] __op other[i]; });                            \
        return self;                                                                             \
    }                                                                                            \
    template <class Self, class T>                                                               \
        requires(Self::Settings.arithmetic.scalar_operations) &&                                 \
                binary_operator_scalar_l<Self, T,                                                \
                                         decltype([](auto const &a, auto const &b)               \
                                                      -> decltype(auto) { return a __op b; }),   \
                                         Self::Settings.arithmetic.implicit_casting>             \
    constexpr Self __op_name(this Self &self, const T &scalar) {                                 \
        static constexpr auto settings =                                                         \
            Self::SimdSettings                                                                   \
                .template get<#__op_name "(this const Self& self, const T& scalar)">();          \
        for_each<Self::size(), settings>([&](std::size_t i) { self[i] = self[i] __op scalar; }); \
        return self;                                                                             \
    }

    DEFINE_OPERATOR(+, add)
    DEFINE_OPERATOR(-, sub)
    DEFINE_OPERATOR(*, mul)
    DEFINE_OPERATOR(/, div)

#undef DEFINE_OPERATOR
}; // arithmetic

struct arithmetic_overloads {
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
    template <class Self, class Other = Self>                                                    \
        requires(Self::size() == Other::size())                                                  \
    constexpr decltype(auto) operator __op(this const Self &self, const Other &other) {          \
        if constexpr (binary_operator_l<Self, Other, decltype([](const auto &a, const auto &b) { \
                                            return a __op b;                                     \
                                        })>) {                                                   \
            static_assert(Self::Settings.arithmetic.implicit_casting ||                          \
                          std::is_same_v<typename Self::base_type, typename Other::base_type>);  \
            Self ret{};                                                                          \
            static constexpr auto settings =                                                     \
                Self::SimdSettings                                                               \
                    .template get<#__op_name "(this const Self& self, const Other& other)">();   \
            for_each<Self::size(), settings>(                                                    \
                [&](std::size_t i) { ret[i] = self[i] __op other[i]; });                         \
            return ret;                                                                          \
        } else if constexpr (binary_operator_r<Self, Other,                                      \
                                               decltype([](const auto &a, const auto &b) {       \
                                                   return a __op b;                              \
                                               })>) {                                            \
            static_assert(Self::Settings.arithmetic.implicit_casting ||                          \
                          std::is_same_v<typename Self::base_type, typename Other::base_type>);  \
            Other ret{};                                                                         \
            static constexpr auto settings =                                                     \
                Other::SimdSettings                                                              \
                    .template get<#__op_name "(this const Self& self, const Other& other)">();   \
            for_each<Self::size(), settings>(                                                    \
                [&](std::size_t i) { ret[i] = self[i] __op other[i]; });                         \
            return ret;                                                                          \
        } else {                                                                                 \
            []<bool v = false> { static_assert(v, "Cannot define operator properly"); }();       \
        }                                                                                        \
    }

    DEFINE_OPERATOR(+, add)
    DEFINE_OPERATOR(-, sub)
    DEFINE_OPERATOR(*, mul)
    DEFINE_OPERATOR(/, div)

#undef DEFINE_OPERATOR
}; // arithmetic_overloads

struct arithmetic_assignment_overloads {
#define DEFINE_OPERATOR(__op, __op_name)                                                         \
    template <class Self, class Other = Self>                                                    \
        requires assign_operator<Self, Other,                                                    \
                                 decltype([](auto &a, auto const &b) -> decltype(auto) {         \
                                     return a __op b;                                            \
                                 })>                                                             \
    constexpr Self &operator __op(this Self & self, const Other & other) {                       \
        static constexpr auto settings =                                                         \
            Self::SimdSettings                                                                   \
                .template get<#__op_name "(this const Self& self, const Other& other)">();       \
        for_each<Self::size(), settings>([&](std::size_t i) { self[i] __op other[i]; });         \
        return self;                                                                             \
    }

    DEFINE_OPERATOR(+=, add_assign)
    DEFINE_OPERATOR(-=, sub_assign)
    DEFINE_OPERATOR(*=, mul_assign)
    DEFINE_OPERATOR(/=, div_assign)

#undef DEFINE_OPERATOR
}; // arithmetic_assignment_overloads
} // namespace details

template <class T, std::size_t N,
          template <class, std::size_t> class S = details::default_storage,
          VectorSettings _Settings = VectorSettings{},
          auto _SimdSettings = SettingsRegistry<SettingsField<"default", SimdSettings{}>>{}>
    requires vec_storage<S, T, N> && (details::validate_settings<_Settings>())
class Vector
    : public std::conditional_t<_Settings.extend_storage, S<T, N>, details::empty_t>,
      public std::conditional_t<_Settings.arithmetic.enabled, details::arithmetic,
                                details::empty_t>,
      public std::conditional_t<_Settings.arithmetic.overloads, details::arithmetic_overloads,
                                details::empty_t>,
      public std::conditional_t<_Settings.arithmetic.assignment,
                                details::arithmetic_assignment_overloads, details::empty_t> {
    [[no_unique_address]]
    std::conditional_t<!_Settings.extend_storage, S<T, N>, details::empty_t> m_storage;

  public:
    constexpr inline static auto SimdSettings = _SimdSettings;
    constexpr inline static VectorSettings Settings = _Settings;
    using base_type = T;
    using Storage = S<T, N>;

    /**
     * \brief Gets the vector's size
     *
     * \return The number of elements
     */
    consteval static std::size_t size() noexcept { return N; }

    constexpr decltype(auto) operator[](this auto &&self, std::size_t i) noexcept {
        if constexpr (Settings.extend_storage) {
            return self.Storage::operator[](i);
        } else {
            return self.m_storage[i];
        }
    }

    friend std::ostream &operator<<(std::ostream &s, const Vector &self) {
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
    constexpr bool operator==(this const Vector &self, const Vector &other) {
        static constexpr auto settings =
            SimdSettings
                .template get<"operator==(this const Vector& self, const Vector& other)">();
        bool equal = true;
        for_each<size(), settings>([&](std::size_t i) { equal &= self[i] == other[i]; });
        return equal;
    }

    constexpr bool operator!=(this const Vector &self, const Vector &other) {
        return !(self == other);
    }
    // }}}

    // {{{ Constructors
    constexpr explicit Vector()
        requires std::default_initializable<Storage>
        : Storage{} {}

    constexpr Vector(const Vector &other)
        requires std::copyable<T>
    {
        if (this == &other) [[unlikely]] {
            throw std::invalid_argument("Attempted to assign to self");
        }

        static constexpr auto settings =
            SimdSettings.template get<"Vector(const Vector& other)">();
        details::for_each<N, settings>(
            [&, this](std::size_t i) { this->operator[](i) = other[i]; });
    }

    constexpr Vector(Vector &&other)
        requires std::default_initializable<Storage> && std::is_move_assignable_v<Storage>
        : Storage{std::exchange(static_cast<Storage &>(other), Storage{})} {
        if (this == &other) [[unlikely]] {
            throw std::invalid_argument("Attempted to assign to self");
        }
    }

    constexpr Vector(std::initializer_list<T> &&list) {
        if (list.size() != size()) [[unlikely]] {
            throw std::invalid_argument("Invalid initializer_list size");
        }

        auto it = list.begin();
        static constexpr auto settings =
            SimdSettings.template get<"Vector(std::initializer_list<T>&& list)">();
        details::for_each<N, settings>(
            [&, this](std::size_t i) { this->operator[](i) = *(it++); });
    }

    constexpr Vector &operator=(this Vector &self, const Vector &other)
        requires std::copyable<T>
    {
        static constexpr auto settings =
            SimdSettings.template get<"operator=(this Vector& self, const Vector& other)">();
        details::for_each<N, settings>([&](std::size_t i) { self[i] = other[i]; });
        return self;
    }

    constexpr ~Vector() {}
    // }}}
}; // Vector
} // namespace vector

#endif // VECTOR_HPP
