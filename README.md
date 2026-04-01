# Neural Networks in C

Low-level neural network implementation in C and POSIX threads parallelism.

## Overview

This project implements from scratch:

- Single Layer Perceptron (SLP)
- Multi-Layer Perceptron (MLP)
- Explicit backpropagation
- Custom BLAS-like linear algebra core
- Parallelization using POSIX threads (pthreads)

The goal is not to build a production-ready framework, but to deeply understand how neural networks operate at a low level:
memory layout, numerical computation, and parallel execution.

## Design Principles

- **C**
- **Row-major memory layout**
- **Manual memory management**
- **Explicit parallelism (pthreads)**
- **No hidden abstractions**

This project prioritizes clarity of execution over abstraction.

## Project Structure

```bash
Neural-Networks-in-C
├── include/
│   ├── matrix.h
│   └── linalg.h
├── src/
│   ├── matrix.c
│   └── linalg.c
├── tests/
│   └── test_*.c
├── build/ # compiled binaries
├── main.c
└── Makefile
```

## Core Components

### Matrix Representation

All data is stored in row-major format:

```c
#define MAT_AT(m, i, j) ((m)->data[(i) * (m)->stride + (j)])
```

![](assets/row-major.png)

This enables:

- Cache-friendly access patterns
- Predictable memory layout
- Efficient parallelization

## Build

```bash
make
```
Compile all tests:

```bash
make test
```

run:

```bash
./build/test_matmul
./build/test_matvec
```

## Test Memory Leak

To use this feature, you need to install valgrind:

```bash
sudo apt-get install valgrind
```

You need to give permission to run the script:

```bash
chmod +x run_valgrind.sh
```

Then run:

```bash
./run_valgrind.sh test_matrix
```

## Goals
- Understand how neural networks work at the lowest level
- Implement backpropagation manually
- Explore performance trade-offs in C
- Learn parallel programming with pthreads

## Future Work

- MLP implementation
- Backpropagation engine
- Batch processing
- Cache-optimized matmul (blocking)
- Gradient checking
- Thread optimization