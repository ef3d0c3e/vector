# Vector - On demand vector type

Vector is a single-header library that provide math vectors to C++.

## Vectors can be structural

By providing you own storage type, you can access a vector's elements easily:
```cpp
template<class T, std::size_t N>
    requires (N == 3)
struct vec3_storage
{
    union {
        T x, y, z;
        T data[N];
    };

    constexpr decltype(auto) operator[](this auto&& self, std::size_t i) {
        return self.data[i];
    }
};

// VectorFeatures::extend_storage is enabled by default!
using v3f = vector::Vec<float, 3, vec3_storage>;

v3f u{7, 3.14, 0.0};
assert(u[1] == u.y);
```

In case you disable `extend_storage`, you can still access the storage using the `_storage` field.

## Vectors are tuple like

Vectors support tuple-like operations, such as `std::get`, `std::tuple_element`, `std::tuple_size` and `std::apply`.

## Vectors are array like

Vectors override the subscript `vec[i]` operator, so you can access their elements by index.

## Vectors are iterable

Vectors provide their own contiguous iterators, accessible via `Vector::iterator`/`Vector::const_iterator`. They also have their reverse (only random access) equivalent: `Vector::reverse_iterator`/`Vector::const_reverse_iterator`. You can access them directly using `std::begin`/`std::end`, or simply `vec.begin()`/`vec.end()`

# Building

Vector uses modern C++ features, it requires C++23 to compile. Vector was tested to work with GCC>=14.2 and Clang>=18.1.8.
According to OpenMP documentation, OpenMP statements cannot appear in `constexpr` functions, however GCC allows it. Therefore the OpenMP features have to be turned off on Clang.
However, the compiler is generally able to efficiently optimize the vector's iterations for `-O>=1`, so OpenMP doesn't make a difference.

# Building the doc

Go to the `docs/` directory and type `doxygen default`.

# License
Vector is licensed under the MIT license.
