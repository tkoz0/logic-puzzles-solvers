#pragma once

#include <assert.h>
#include <stdlib.h>

typedef struct
{
    void **array; // array data
    size_t len; // length of array
    size_t alloc; // allocated length
}
ptrarr_t;

static inline void ptrarr_init(ptrarr_t *arr)
{
    arr->array = NULL;
    arr->len = 0;
    arr->alloc = 0;
}

static inline void ptrarr_clear(ptrarr_t *arr)
{
    free(arr->array);
    arr->array = NULL;
    arr->len = 0;
    arr->alloc = 0;
}

static inline void ptrarr_push(ptrarr_t *arr, void *ptr)
{
    if (arr->len == arr->alloc) // grow array
    {
        arr->alloc += (arr->alloc >> 2) + 1; // growth factor ~1.25
        arr->array = (void**)realloc(arr->array,arr->alloc*sizeof(*arr->array));
        assert(arr->array);
    }
    arr->array[arr->len++] = ptr;
}
