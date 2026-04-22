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
import org.elasticsearch.threadpool.ThreadPool;

import java.io.Closeable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;
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
            : Runtime.getRuntime().availableProcessors();
    }

    /**
     * Submits an HNSW graph build task for background execution.
     *
     * @param task the graph construction runnable
     * @param estimatedCostBytes estimated memory cost (used for backpressure decisions)
     * @return a future that completes when the graph is built
     */
    public CompletableFuture<Void> submitBuildTask(Runnable task, long estimatedCostBytes) {
        if (closed) {
            return CompletableFuture.failedFuture(new RejectedExecutionException("VectorBuildExecutorService is closed"));
        }

        pendingTasks.incrementAndGet();
        long queueStartNanos = System.nanoTime();

        CompletableFuture<Void> future = new CompletableFuture<>();
        try {
            executorService.execute(() -> {
                pendingTasks.decrementAndGet();
                runningTasks.incrementAndGet();
                long queueTimeNanos = System.nanoTime() - queueStartNanos;
                totalQueueTimeNanos.addAndGet(queueTimeNanos);
                long buildStartNanos = System.nanoTime();
                try {
                    task.run();
                    future.complete(null);
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
            future.completeExceptionally(e);
        }

        return future;
    }

    /**
     * Returns true if indexing should be throttled due to too many pending graph builds.
     */
    public boolean shouldThrottleIndexing() {
        return pendingTasks.get() + runningTasks.get() > maxConcurrentBuilds * 3;
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
            : Runtime.getRuntime().availableProcessors();
    }

    @Override
    public void close() {
        closed = true;
    }
}
