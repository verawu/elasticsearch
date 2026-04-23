/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.engine;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.threadpool.TestThreadPool;
import org.elasticsearch.threadpool.ThreadPool;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.greaterThanOrEqualTo;
import static org.hamcrest.Matchers.instanceOf;
import static org.hamcrest.Matchers.is;

public class VectorBuildExecutorServiceTests extends ESTestCase {

    private TestThreadPool threadPool;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        threadPool = new TestThreadPool("vector-build-test", Settings.EMPTY);
    }

    @Override
    public void tearDown() throws Exception {
        ThreadPool.terminate(threadPool, 30, TimeUnit.SECONDS);
        super.tearDown();
    }

    public void testSubmitAndComplete() throws Exception {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        CompletableFuture<String> future = service.submitBuildTask(() -> "hello", 1024);
        assertThat(future.get(10, TimeUnit.SECONDS), equalTo("hello"));
        assertBusy(() -> {
            assertThat(service.getCompletedTasks(), equalTo(1L));
            assertThat(service.getTotalBuildTimeNanos(), greaterThan(0L));
        });
        service.close();
    }

    public void testSubmitWhenClosed() {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        service.close();
        CompletableFuture<String> future = service.submitBuildTask(() -> "should fail", 1024);
        assertTrue(future.isCompletedExceptionally());
        CompletionException ce = expectThrows(CompletionException.class, future::join);
        assertThat(ce.getCause(), instanceOf(RejectedExecutionException.class));
    }

    public void testConcurrentSubmissions() throws Exception {
        int taskCount = 20;
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        Semaphore gate = new Semaphore(0);
        CountDownLatch allStarted = new CountDownLatch(taskCount);

        List<CompletableFuture<Integer>> futures = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            final int idx = i;
            futures.add(service.submitBuildTask(() -> {
                allStarted.countDown();
                gate.acquire();
                return idx;
            }, 1024));
        }

        assertBusy(() -> assertThat(service.getPendingTasks() + service.getRunningTasks(), greaterThan(0)));

        gate.release(taskCount);
        for (int i = 0; i < taskCount; i++) {
            futures.get(i).get(30, TimeUnit.SECONDS);
        }

        assertBusy(() -> assertThat(service.getCompletedTasks(), equalTo((long) taskCount)));
        assertThat(service.getPendingTasks(), equalTo(0));
        assertThat(service.getRunningTasks(), equalTo(0));
        service.close();
    }

    public void testShouldThrottleIndexing() throws Exception {
        int maxConcurrent = 2;
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, maxConcurrent);
        int threshold = maxConcurrent * 3;
        int taskCount = threshold + 5;

        Semaphore gate = new Semaphore(0);
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            futures.add(service.submitBuildTask(() -> {
                gate.acquire();
                return null;
            }, 1024));
        }

        assertBusy(() -> assertTrue("should throttle when queue is saturated", service.shouldThrottleIndexing()));

        gate.release(taskCount);
        for (CompletableFuture<Void> f : futures) {
            f.get(30, TimeUnit.SECONDS);
        }

        assertBusy(() -> assertFalse("should not throttle when queue is drained", service.shouldThrottleIndexing()));
        service.close();
    }

    public void testStatsTracking() throws Exception {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        int taskCount = 5;
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            futures.add(service.submitBuildTask(() -> {
                Thread.sleep(1);
                return null;
            }, 2048));
        }
        for (CompletableFuture<Void> f : futures) {
            f.get(10, TimeUnit.SECONDS);
        }

        assertBusy(() -> {
            assertThat(service.getCompletedTasks(), equalTo((long) taskCount));
            assertThat(service.getTotalBuildTimeNanos(), greaterThan(0L));
            assertThat(service.getTotalQueueTimeNanos(), greaterThanOrEqualTo(0L));
        });
        service.close();
    }

    public void testSetMaxConcurrentBuilds() throws Exception {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 1);

        Semaphore gate = new Semaphore(0);
        int taskCount = 10;
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (int i = 0; i < taskCount; i++) {
            futures.add(service.submitBuildTask(() -> {
                gate.acquire();
                return null;
            }, 1024));
        }

        assertBusy(() -> assertTrue(service.shouldThrottleIndexing()));

        service.setMaxConcurrentBuilds(100);
        assertFalse("after raising max, should no longer throttle", service.shouldThrottleIndexing());

        gate.release(taskCount);
        for (CompletableFuture<Void> f : futures) {
            f.get(30, TimeUnit.SECONDS);
        }
        service.close();
    }

    public void testTaskFailurePropagation() throws Exception {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        RuntimeException expected = new RuntimeException("build failed");
        CompletableFuture<Void> future = service.submitBuildTask(() -> { throw expected; }, 1024);

        CompletionException ce = expectThrows(CompletionException.class, future::join);
        assertThat(ce.getCause(), is(expected));
        assertBusy(() -> assertThat(service.getCompletedTasks(), equalTo(1L)));
        service.close();
    }

    public void testCloseAwaitsInFlightTasks() throws Exception {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        CountDownLatch taskStarted = new CountDownLatch(1);
        Semaphore gate = new Semaphore(0);

        CompletableFuture<String> future = service.submitBuildTask(() -> {
            taskStarted.countDown();
            gate.acquire();
            return "done";
        }, 1024);

        assertTrue(taskStarted.await(10, TimeUnit.SECONDS));

        Thread closeThread = new Thread(service::close);
        closeThread.start();

        Thread.sleep(100);
        assertTrue("close should block while task is in-flight", closeThread.isAlive());

        gate.release();
        closeThread.join(10_000);
        assertFalse("close should have completed", closeThread.isAlive());
        assertThat(future.get(1, TimeUnit.SECONDS), equalTo("done"));
    }

    public void testAwaitPendingTasksReturnsImmediatelyWhenEmpty() {
        VectorBuildExecutorService service = new VectorBuildExecutorService(threadPool, 4);
        assertTrue(service.awaitPendingTasks(1, TimeUnit.MILLISECONDS));
        service.close();
    }
}
