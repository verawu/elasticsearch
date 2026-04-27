/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.engine;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Setting.Property;
import org.elasticsearch.index.shard.DenseVectorStats;
import org.elasticsearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Node-level service that manages background HNSW graph construction tasks.
 * Deferred vector index building submits graph construction work to this service,
 * which executes on the {@link ThreadPool.Names#VECTOR_BUILD} thread pool.
 */
public class VectorBuildExecutorService implements Closeable {

    private static final Logger logger = LogManager.getLogger(VectorBuildExecutorService.class);

    public static final Setting<Boolean> DEFERRED_VECTOR_BUILD_ENABLED_SETTING = Setting.boolSetting(
        "index.vector_build.deferred",
        false,
        Property.IndexScope,
        Property.Dynamic
    );

    public static final Setting<Integer> MAX_CONCURRENT_BUILDS_SETTING = Setting.intSetting(
        "indices.vector_build.max_concurrent",
        -1,
        -1,
        Property.NodeScope,
        Property.Dynamic
    );

    private final ExecutorService executorService;
    private final Set<CompletableFuture<?>> inFlightFutures = ConcurrentHashMap.newKeySet();
    private final ConcurrentHashMap<String, AtomicInteger> perIndexInFlight = new ConcurrentHashMap<>();
    private final AtomicInteger pendingTasks = new AtomicInteger();
    private final AtomicInteger runningTasks = new AtomicInteger();
    private final AtomicLong completedTasks = new AtomicLong();
    private final AtomicLong totalBuildTimeNanos = new AtomicLong();
    private final AtomicLong totalQueueTimeNanos = new AtomicLong();
    private volatile int maxConcurrentBuilds;
    private volatile boolean closed;

    public VectorBuildExecutorService(ThreadPool threadPool, int maxConcurrentBuilds) {
        this.executorService = threadPool.executor(ThreadPool.Names.VECTOR_BUILD);
        this.maxConcurrentBuilds = maxConcurrentBuilds > 0
            ? maxConcurrentBuilds
            : defaultMaxConcurrent();
    }

    /**
     * Submits an HNSW graph build task for background execution.
     *
     * @param indexName the index this build belongs to (used for per-index fair scheduling)
     * @param task the graph construction callable
     * @param estimatedCostBytes estimated memory cost (used for backpressure decisions)
     * @return a future that completes with the callable's result when the graph is built
     */
    public <T> CompletableFuture<T> submitBuildTask(String indexName, Callable<T> task, long estimatedCostBytes) {
        if (closed) {
            return CompletableFuture.failedFuture(new RejectedExecutionException("VectorBuildExecutorService is closed"));
        }

        pendingTasks.incrementAndGet();
        perIndexInFlight.computeIfAbsent(indexName, k -> new AtomicInteger()).incrementAndGet();
        long queueStartNanos = System.nanoTime();

        CompletableFuture<T> future = new CompletableFuture<>();
        inFlightFutures.add(future);
        future.whenComplete((r, t) -> {
            inFlightFutures.remove(future);
            AtomicInteger counter = perIndexInFlight.get(indexName);
            if (counter != null && counter.decrementAndGet() <= 0) {
                perIndexInFlight.remove(indexName, counter);
            }
        });
        try {
            executorService.execute(() -> {
                pendingTasks.decrementAndGet();
                runningTasks.incrementAndGet();
                long queueTimeNanos = System.nanoTime() - queueStartNanos;
                totalQueueTimeNanos.addAndGet(queueTimeNanos);
                long buildStartNanos = System.nanoTime();
                try {
                    future.complete(task.call());
                } catch (Exception e) {
                    logger.warn("HNSW graph build failed", e);
                    future.completeExceptionally(e);
                } finally {
                    runningTasks.decrementAndGet();
                    completedTasks.incrementAndGet();
                    totalBuildTimeNanos.addAndGet(System.nanoTime() - buildStartNanos);
                }
            });
        } catch (RejectedExecutionException e) {
            pendingTasks.decrementAndGet();
            AtomicInteger counter = perIndexInFlight.get(indexName);
            if (counter != null && counter.decrementAndGet() <= 0) {
                perIndexInFlight.remove(indexName, counter);
            }
            future.completeExceptionally(e);
        }

        return future;
    }

    /**
     * Returns true if indexing should be throttled for the given index.
     * Throttles when either: (a) the global queue is saturated, or
     * (b) this index alone exceeds its fair share of the build capacity.
     */
    public boolean shouldThrottleIndexing(String indexName) {
        int globalThreshold = maxConcurrentBuilds * 3;
        int totalInFlight = pendingTasks.get() + runningTasks.get();
        if (totalInFlight > globalThreshold) {
            return true;
        }
        AtomicInteger counter = perIndexInFlight.get(indexName);
        int indexInFlight = counter != null ? counter.get() : 0;
        int activeIndices = Math.max(1, perIndexInFlight.size());
        int fairShare = Math.max(1, globalThreshold / activeIndices);
        return indexInFlight > fairShare;
    }

    public DenseVectorStats.VectorBuildStats stats() {
        return new DenseVectorStats.VectorBuildStats(
            pendingTasks.get(),
            runningTasks.get(),
            completedTasks.get(),
            TimeUnit.NANOSECONDS.toMillis(totalBuildTimeNanos.get()),
            TimeUnit.NANOSECONDS.toMillis(totalQueueTimeNanos.get())
        );
    }

    /**
     * Waits for all currently in-flight graph build tasks to complete.
     * @return true if all tasks completed within the timeout, false if timed out
     */
    public boolean awaitPendingTasks(long timeout, TimeUnit unit) {
        if (inFlightFutures.isEmpty()) {
            return true;
        }
        try {
            CompletableFuture.allOf(inFlightFutures.toArray(new CompletableFuture<?>[0])).get(timeout, unit);
            return true;
        } catch (TimeoutException e) {
            logger.warn("Timed out waiting for {} in-flight vector build tasks", inFlightFutures.size());
            return false;
        } catch (Exception e) {
            return true;
        }
    }

    public ExecutorService getExecutorService() {
        return executorService;
    }

    public int getPendingTasks() {
        return pendingTasks.get();
    }

    public int getRunningTasks() {
        return runningTasks.get();
    }

    public long getCompletedTasks() {
        return completedTasks.get();
    }

    public long getTotalBuildTimeNanos() {
        return totalBuildTimeNanos.get();
    }

    public long getTotalQueueTimeNanos() {
        return totalQueueTimeNanos.get();
    }

    public void setMaxConcurrentBuilds(int maxConcurrentBuilds) {
        this.maxConcurrentBuilds = maxConcurrentBuilds > 0
            ? maxConcurrentBuilds
            : defaultMaxConcurrent();
    }

    static int defaultMaxConcurrent() {
        int procs = Runtime.getRuntime().availableProcessors();
        return Math.max(1, Math.min((procs + 1) / 2, 5));
    }

    @Override
    public void close() {
        closed = true;
        awaitPendingTasks(60, TimeUnit.SECONDS);
    }
}
