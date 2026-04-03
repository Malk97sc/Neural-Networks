#include <stdlib.h>
#include <pthread.h>

#include "thread_pool.h"

struct ThreadPool {
    pthread_t *threads; //all workers
    pthread_mutex_t mutex;
    pthread_cond_t work_ready; //workers
    pthread_cond_t work_done; //main thread

    task_fn fn; //function to execute
    void **args;
    int n_tasks;
    int task_idx; //next task index to pick
    int n_done;

    int n_threads;
    int shutdown; //exit flag
};

static void *worker_loop(void *arg){
    ThreadPool *pool = (ThreadPool *)arg;
    int idx;

    while(1){
        pthread_mutex_lock(&pool->mutex);

        while(pool->task_idx >= pool->n_tasks && !pool->shutdown){
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        }

        if(pool->shutdown){
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }

        idx = pool->task_idx++;
        pthread_mutex_unlock(&pool->mutex);

        pool->fn(pool->args[idx]);

        pthread_mutex_lock(&pool->mutex);
        pool->n_done++;
        if(pool->n_done == pool->n_tasks){
            pthread_cond_signal(&pool->work_done);
        }            
        pthread_mutex_unlock(&pool->mutex);
    }
}

ThreadPool *thread_pool_create(int n_threads){
    ThreadPool *pool = (ThreadPool *) malloc(sizeof(ThreadPool));

    pool->threads = (pthread_t *) malloc(n_threads * sizeof(pthread_t));
    pool->n_threads = n_threads;
    pool->fn = NULL;
    pool->args = NULL;
    pool->n_tasks = 0;
    pool->task_idx = 0;
    pool->n_done = 0;
    pool->shutdown = 0;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->work_ready, NULL);
    pthread_cond_init(&pool->work_done, NULL);

    for(int i = 0; i < n_threads; i++){
        pthread_create(&pool->threads[i], NULL, worker_loop, pool);
    }        

    return pool;
}

void thread_pool_submit(ThreadPool *pool, task_fn fn, void **args, int n_tasks){
    pthread_mutex_lock(&pool->mutex);

    pool->fn = fn;
    pool->args = args;
    pool->n_tasks = n_tasks;
    pool->task_idx = 0;
    pool->n_done = 0;

    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
}

void thread_pool_wait(ThreadPool *pool){ //manual barrier
    pthread_mutex_lock(&pool->mutex);
    while(pool->n_done < pool->n_tasks){ 
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }        
    pthread_mutex_unlock(&pool->mutex);
}

void thread_pool_destroy(ThreadPool *pool){
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    for(int i = 0; i < pool->n_threads; i++){
        pthread_join(pool->threads[i], NULL);
    }        

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->work_ready);
    pthread_cond_destroy(&pool->work_done);

    free(pool->threads);
    free(pool);
}
