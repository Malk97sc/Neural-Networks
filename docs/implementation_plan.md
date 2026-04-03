# Thread Pool: Minimalist BLAS-Style Implementation

The root cause of the lack of speedup is that every call to `matvec_parallel_impl`, `matmul_parallel_impl`, etc. performs a full `pthread_create` + `pthread_join` cycle. This syscall overhead dominates the actual computation for most matrix sizes.

The solution is a **persistent thread pool**: workers are created once at startup (`runtime_init`) and sleep on a condition variable when idle. The main thread wakes them, they execute a task, and go back to sleep. No thread is ever destroyed until `runtime_destroy` is called.

## Proposed Changes

---

### Thread Pool Core

#### [NEW] [thread_pool.h](file:///home/dexx/SharedDisk/Programming/NN-project/include/thread_pool.h)

Minimal public API:

```c
typedef void (*task_fn)(void *arg);

typedef struct ThreadPool ThreadPool;

ThreadPool *thread_pool_create(int n_threads);
void        thread_pool_submit(ThreadPool *pool, task_fn fn, void **args, int n_tasks);
void        thread_pool_wait(ThreadPool *pool);
void        thread_pool_destroy(ThreadPool *pool);
```

- `thread_pool_submit`: dispatches `n_tasks` work units. Each worker picks an `args[i]` and calls `fn(args[i])`.
- `thread_pool_wait`: blocks until all submitted tasks are done (internal barrier).

#### [NEW] [thread_pool.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/thread_pool.c)

Internal design:

```
ThreadPool {
    pthread_t      *threads;       // persistent workers
    pthread_mutex_t mutex;
    pthread_cond_t  work_ready;    // signal workers
    pthread_cond_t  work_done;     // signal main thread
    task_fn         fn;            // current task function
    void          **args;          // array of per-worker args
    int             n_tasks;       // total tasks dispatched
    int             n_done;        // tasks completed
    int             n_threads;
    int             shutdown;      // exit flag
}
```

Workers loop: lock mutex -> wait on `work_ready` -> execute assigned task -> increment `n_done` -> signal `work_done` -> repeat.

---

### Runtime Integration

#### [MODIFY] [runtime.h](file:///home/dexx/SharedDisk/Programming/NN-project/include/runtime.h)

- Add `ThreadPool *pool` field to `RuntimeConfig`.
- Add `void runtime_destroy(void)` declaration.

#### [MODIFY] [runtime.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/runtime.c)

- `runtime_init`: call `thread_pool_create(n_threads)` and store in `config.pool`.
- `runtime_destroy`: call `thread_pool_destroy(config.pool)`.

---

### Parallel Operations Refactoring

All four `_impl` functions in `src/parallel/` currently do:
1. Declare `pthread_t threads[n_threads]`
2. Call `pthread_create` for each thread
3. Call `pthread_join` for each thread

They will be refactored to:
1. Populate an `args[]` array with per-task structs (same as before)
2. Call `thread_pool_submit(pool, worker_fn, args, n_threads)`
3. Call `thread_pool_wait(pool)`

The `worker` functions themselves remain unchanged; only the scheduler changes.

#### [MODIFY] [matvec_parallel.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/parallel/matvec_parallel.c)
#### [MODIFY] [matmul_parallel.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/parallel/matmul_parallel.c)
#### [MODIFY] [mat_add_rowwise_parallel.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/parallel/mat_add_rowwise_parallel.c)
#### [MODIFY] [mat_apply_parallel.c](file:///home/dexx/SharedDisk/Programming/NN-project/src/parallel/mat_apply_parallel.c)

Each file needs access to the pool via `runtime_get()->pool`.

---

### Build

#### [MODIFY] [Makefile](file:///home/dexx/SharedDisk/Programming/NN-project/Makefile)

Add `src/thread_pool.c` to the `SRC` variable so it compiles and links automatically.

---

### Tests Cleanup

#### [MODIFY] test files that call `runtime_init`

Add `runtime_destroy()` at the end of `main` in `test_runtime.c` and `test_performance.c` to properly shut down the pool and allow Valgrind to confirm 0 leaks.

---

## Verification Plan

### Automated Tests

```bash
# Compile everything
make clean && make test

# Correctness: all unit tests must still pass
./build/test_linalg
./build/test_linalg_parallel
./build/test_matrix
./build/test_runtime

# Performance: speedup should now be measurable
./build/test_performance

# Memory: must report 0 errors and 0 leaks
./run_valgrind.sh test_runtime
./run_valgrind.sh test_linalg
```

### Expected Results
- All `assert` tests pass without modification.
- `test_performance` shows a measurable speedup (`speedup > 1.0x`) for the large matrix sizes.
- Valgrind reports `ERROR SUMMARY: 0` and `All heap blocks were freed`.
