#include <functional>
#include <iostream>
#include <chrono>
#include "settings.hpp"
#include "vector.hpp"

int main()
{
	using settings = vector::SettingsRegistry<
		vector::SettingsField<"default", vector::SimdSettings::NONE>,
		vector::SettingsField<"operator=(this Vector& self, const Vector& other)", vector::SimdSettings::NONE>
	>;
	/*
	using Vec = vector::Vector<float, 16, std::array, vector::VectorFeatures{}, settings{}>;

	Vec v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};//{1,2,3, 4};
	//std::cout << std::format("{:5f}", v);
	
	std::cout << std::tuple_size_v<Vec>;
	*/

	using Vec = vector::Vector<int, 3, std::array, vector::VectorFeatures{}, settings{}>;
	const Vec u{1, 2, 3};

	const auto& [x, y, z] = u;

	std::cout << x << " " << y << " " << z;
	

/*
	using clock_t = std::chrono::high_resolution_clock;
	double total = 0;

	volatile std::size_t count = 10'000'000;
	for (std::size_t i = 0; i < 100; ++i)
	{
		const auto start = clock_t::now();
		auto y = Vec{};
		for (std::size_t i = 0; i < count; ++i) {
			y = y + v;
		}
		const auto end = clock_t::now();

		total += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
		std::cout << y << " : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start) << std::endl;
	}

	std::cout << "avg: " << total/100 << "µs\n";
	std::cout << "ops: " << total/100/count*1000 << "ns";
	*/

	return 0;
}
