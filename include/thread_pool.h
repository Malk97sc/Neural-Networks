#ifndef THREAD_POOL_H
#define THREAD_POOL_H

typedef void (*task_fn)(void *arg);

typedef struct ThreadPool ThreadPool;

ThreadPool *thread_pool_create(int n_threads);
void thread_pool_submit(ThreadPool *pool, task_fn fn, void **args, int n_tasks);
void thread_pool_wait(ThreadPool *pool);
void thread_pool_destroy(ThreadPool *pool);

#endif
