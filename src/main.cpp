#include <functional>
#include <iostream>
#include <chrono>
#include "vector.hpp"

int main()
{
	vector::Vector<double, 4> v{1,2,3, 4};

	using clock_t = std::chrono::high_resolution_clock;
	volatile std::size_t count = 10'000'000;

	double total = 0;
	for (std::size_t i = 0; i < 100; ++i)
	{
		const auto start = clock_t::now();
		auto y = vector::Vector<double, 4>{};
		for (std::size_t i = 0; i < count; ++i) {
			y = y + v;
		}
		const auto end = clock_t::now();

		total += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
		std::cout << y << " : " << std::chrono::duration_cast<std::chrono::microseconds>(end-start) << std::endl;
	}

	std::cout << "avg:" << total/100 << "µs";


	//v[2] = 7;
	return 0;
}
