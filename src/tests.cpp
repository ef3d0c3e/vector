#include "catch2/generators/catch_generators_random.hpp"
#include "catch2/internal/catch_random_seed_generation.hpp"
#include "vector.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

TEST_CASE("Basic operators", "[v4]")
{
	[]<class... Ts>(std::tuple<Ts...>) {
		(([]<class T> {
			 using v4 = vector::Vector<T, 4>;

			 const v4 a{ 1, 2, 3, 4 };
			 const v4 b{ 2, 8, 16, 32 };

			 REQUIRE(a + b == a.clone().add(b));
			 REQUIRE(a - b == a.clone().sub(b));
			 REQUIRE(a * b == a.clone().mul(b));
			 REQUIRE(a / b == a.clone().div(b));

			 // Scalar
			 REQUIRE(a + 5 == a.clone().add(5));
			 REQUIRE(a - 5 == a.clone().sub(5));
			 REQUIRE(a * 5 == a.clone().mul(5));
			 REQUIRE(a / 5 == a.clone().div(5));
		 }.template operator()<Ts>()),
		 ...);
	}(std::tuple<int, long, float, double>{});
}

/// @brief Structural Vec3 storage
template<class T, std::size_t N>
    requires(N == 3)
struct vec3_storage
{
	union
	{
		T data[3];
		T x, y, z;
	};

	constexpr decltype(auto) operator[](this auto&& self, std::size_t i) { return (self.data[i]); }
};

/// @brief Non structural Vec3 storage
template<class T, std::size_t N>
    requires(N == 3)
class vec3_storage_ns
{
	union
	{
		T data[3];
		T x, y, z;
	};

public:
	constexpr decltype(auto) operator[](this auto&& self, std::size_t i) { return (self.data[i]); }
};

TEST_CASE("Custom storage", "[v3]")
{

	[]<class... Ts>(std::tuple<Ts...>) {
		(([]<class T> {
			 {
				 using v3s = vector::Vector<T, 3, vec3_storage>;
				 static_assert(sizeof(v3s) == sizeof(T) * 3);

				 const v3s a{ 1, 2, 3 };
				 const v3s b{ 2, 8, 16 };

				 REQUIRE(a + b == a.clone().add(b));
				 REQUIRE(a - b == a.clone().sub(b));
				 REQUIRE(a * b == a.clone().mul(b));
				 REQUIRE(a / b == a.clone().div(b));

				 // Scalar
				 REQUIRE(a + 5 == a.clone().add(5));
				 REQUIRE(a - 5 == a.clone().sub(5));
				 REQUIRE(a * 5 == a.clone().mul(5));
				 REQUIRE(a / 5 == a.clone().div(5));
			 }

			 {
				 using v3ns = vector::Vector<T, 3, vec3_storage_ns>;
				 static_assert(sizeof(v3ns) == sizeof(T) * 3);

				 const v3ns a{ 1, 2, 3 };
				 const v3ns b{ 2, 8, 16 };

				 REQUIRE(a + b == a.clone().add(b));
				 REQUIRE(a - b == a.clone().sub(b));
				 REQUIRE(a * b == a.clone().mul(b));
				 REQUIRE(a / b == a.clone().div(b));

				 // Scalar
				 REQUIRE(a + 5 == a.clone().add(5));
				 REQUIRE(a - 5 == a.clone().sub(5));
				 REQUIRE(a * 5 == a.clone().mul(5));
				 REQUIRE(a / 5 == a.clone().div(5));
			 }
		 }.template operator()<Ts>()),
		 ...);
	}(std::tuple<int, long, float, double>{});
}

TEST_CASE("Convenience", "Features")
{
	REQUIRE(vector::Vector<int, 2>{ 3, 4 }.dist_squared() == 25);
	REQUIRE(vector::Vector<int, 2>{ 3, 4 }.dist<float>() == 5.f);
	REQUIRE(vector::Vector<float, 2>{ 3, 4 }.dist_squared() == 25.f);
	REQUIRE(vector::Vector<float, 2>{ 3, 4 }.dist<float>() == 5.f);

	auto gen = Catch::Generators::RandomFloatingGenerator<double>(-1.0, 1.0, time(NULL));
	for (auto _ : std::ranges::iota_view{ 0, 100 }) {
		std::array<double, 4> data{};
		double distsq = 0.0;
		for (auto& x : data) {
			gen.next();
			x = gen.get();
			distsq += x * x;
		}

		auto vec = vector::Vector<double, 4>{ std::move(data) };
		REQUIRE(vec.dist_squared() == distsq);
		REQUIRE(vec.dist<double>() - std::sqrt(distsq) < 1E-10);
		REQUIRE(vec.dot(vector::Vector<double, 4>{1,0,0,0}) == vec[0]);
		REQUIRE(vec.dot(vector::Vector<double, 4>{0,1,0,0}) == vec[1]);
		REQUIRE(vec.dot(vector::Vector<double, 4>{0,0,1,0}) == vec[2]);
		REQUIRE(vec.dot(vector::Vector<double, 4>{0,0,0,1}) == vec[3]);
	}

}

/// @brief Features tests
TEST_CASE("Iterators", "Features")
{
	[]<auto... Fs>(std::integral_constant<vector::VectorFeatures, Fs>...){
		(([]<auto F>{
			using v4 = vector::Vector<std::size_t, 4, std::array, F>;

			static_assert(std::ranges::contiguous_range<v4>);

			// Iterators concepts
			static_assert(std::contiguous_iterator<typename v4::iterator>);
			static_assert(std::contiguous_iterator<typename v4::const_iterator>);
			static_assert(std::random_access_iterator<typename v4::reverse_iterator>);
			static_assert(std::random_access_iterator<typename v4::const_reverse_iterator>);

			const v4 v{0,1,2,3};

			// Forward it
			std::size_t i{0};
			for (const auto& x : v)
			{
				REQUIRE(i == x);
				++i;
			}
			for ( auto i : std::ranges::iota_view{0uz, 4uz})
			{
				REQUIRE(std::begin(v)[i] == i);
			}

			// Const it
			i = 0;
			for (auto it = std::cbegin(v); it != std::cend(v); ++it)
			{
				REQUIRE(*it == i);
				++i;
			}
			for ( auto i : std::ranges::iota_view{0uz, 4uz})
			{
				REQUIRE(std::cbegin(v)[i] == i);
			}

			// Reverse it
			i = 3;
			for (auto it = std::rbegin(v); it != std::rend(v); ++it)
			{
				REQUIRE(*it == i);
				--i;
			}
			for ( auto i : std::ranges::iota_view{0uz, 4uz})
			{
				REQUIRE(std::rbegin(v)[i] == 3-i);
			}


			// Const reverse it
			i = 3;
			for (auto it = std::crbegin(v); it != std::crend(v); ++it)
			{
				REQUIRE(*it == i);
				--i;
			}
			for ( auto i : std::ranges::iota_view{0uz, 4uz})
			{
				REQUIRE(std::crbegin(v)[i] == 3-i);
			}

		}.template operator()<Fs>()),...);
	}(
		std::integral_constant<vector::VectorFeatures, vector::VectorFeatures{
			.iterators = {
				.enabled = true,
				.use_storage = false,
			},
			.extend_storage = true,
		}>{},
		std::integral_constant<vector::VectorFeatures, vector::VectorFeatures{
			.iterators = {
				.enabled = true,
				.use_storage = false,
			},
			.extend_storage = false,
		}>{},
		std::integral_constant<vector::VectorFeatures, vector::VectorFeatures{
			.iterators = {
				.enabled = true,
				.use_storage = true,
			},
			.extend_storage = true,
		}>{},
		std::integral_constant<vector::VectorFeatures, vector::VectorFeatures{
			.iterators = {
				.enabled = true,
				.use_storage = true,
			},
			.extend_storage = false,
		}>{}
	);
}

TEST_CASE("Tuple-like", "Features")
{
	[]<auto... Fs>(std::integral_constant<vector::VectorFeatures, Fs>...) {
		(([]<auto F> {
		  	 // Structural storage
			 {
				using v3 = vector::Vector<std::size_t, 3, std::array, F>;
				
				v3 v{0,1,2};
				[&]<auto... _>(std::index_sequence<_...>)
				{
					(([&]<std::size_t i>{
						REQUIRE(get<i>(v) == i);
					}.template operator()<_>()), ...);
				}(std::make_index_sequence<3>{});

				auto&& [x, y, z] = v;
				REQUIRE(v3{x, y, z} == v);
			 }

		  	 // Non-structural storage
			 {
				using v3 = vector::Vector<std::size_t, 3, vec3_storage_ns, F>;
				
				v3 v{0,1,2};
				[&]<auto... _>(std::index_sequence<_...>)
				{
					(([&]<std::size_t i>{
						REQUIRE(get<i>(v) == i);
					}.template operator()<_>()), ...);
				}(std::make_index_sequence<3>{});

				auto&& [x, y, z] = v;
				REQUIRE(v3{x, y, z} == v);
			 }
		 }.template operator()<Fs>()),
		 ...);
	}(std::integral_constant<vector::VectorFeatures,
	                         vector::VectorFeatures{
							 .tuple = {
								.size = true,
								.element = true,
							 	.get = true,
							 },
	                           .extend_storage = true,
	                         }>{},
	  std::integral_constant<vector::VectorFeatures,
	                         vector::VectorFeatures{
							 .tuple = {
								.size = true,
								.element = true,
							 	.get = true,
							 },
	                           .extend_storage = false,
	                         }>{});
}

TEST_CASE("Formatting", "Features")
{
	[]<auto... Fs>(std::integral_constant<vector::VectorFeatures, Fs>...){
		(([]<auto F>{
			using v4 = vector::Vector<float, 4, std::array, F>;
	
			std::stringstream ss;
			v4 v{0.5,1.5,2.3,3.0};

			ss << v;
			REQUIRE(ss.str() == "(0.5, 1.5, 2.3, 3)");

			ss.str({});

			ss << std::format("{:.2f}", v);
			REQUIRE(ss.str() == "(0.50, 1.50, 2.30, 3.00)");

		}.template operator()<Fs>()),...);
	}(
		std::integral_constant<vector::VectorFeatures, vector::VectorFeatures{
			.formatting = {
				.ostream = true,
				.format = true,
			},
		}>{}
	);
}
