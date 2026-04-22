/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && !defined(__clang__)
#define EXPORT __attribute__((externally_visible,visibility("default")))
#elif __clang__
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int vec_caps();

EXPORT int32_t dot7u(int8_t* a, int8_t* b, size_t dims);

EXPORT int32_t sqr7u(int8_t *a, int8_t *b, size_t length);

EXPORT void dot7u_batch(int8_t* query, int8_t* docs, size_t dims, size_t stride, size_t count, int32_t* results);

EXPORT void sqr7u_batch(int8_t* query, int8_t* docs, size_t dims, size_t stride, size_t count, int32_t* results);
