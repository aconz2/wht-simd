Implements an in-place [Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform) for array size 16,32,64,128 with AVX2 SIMD.

Timing numbers are pretty variable since I'm subtracting loop overhead out (see code) but on 5950x Zen3 I get 3 ns compared to 97 ns. The generated code looks nice.

License: public domain
