Implements an in-place [Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform) for array size 32 with AVX2 SIMD.

Timing numbers are pretty variable since I'm subtracting loop overhead out (see code) but on 5950x Zen3 I get 1.8 - 2.5 ns compared to 21 - 25 ns. The generated code looks nice.

License: public domain
