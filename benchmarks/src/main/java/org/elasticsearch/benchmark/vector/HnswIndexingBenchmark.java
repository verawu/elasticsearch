/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene912.Lucene912Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.ESHnswVectorsFormat;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * Benchmarks HNSW vector indexing comparing inline vs deferred graph building.
 *
 * <p>Two benchmark methods:
 * <ul>
 *   <li>{@code addDocumentsOnly} — measures only addDocument() time (no commit/flush). Deferred is
 *       O(1)/doc (flat vector store), inline is O(log N * beamWidth)/doc (graph insertion). This
 *       isolates the indexing-thread cost. Divide numVectors by the result to get docs/sec throughput.</li>
 *   <li>{@code indexAndFlush} — measures full cycle: add all docs + commit (flush + graph build) + kNN
 *       search. Total wall-clock is comparable since graph work is done either way; deferred mode
 *       trades faster addDocument() for a heavier commit.</li>
 * </ul>
 *
 * <p>Run with: {@code ./gradlew -p benchmarks run --args 'HnswIndexingBenchmark'}
 */
@Fork(value = 1)
@Warmup(iterations = 1, time = 1, batchSize = 1)
@Measurement(iterations = 3, time = 1, batchSize = 1)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class HnswIndexingBenchmark {

    static {
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.setNodeName("bench");
        LogConfigurator.configureESLogging();
    }

    @Param({ "inline", "deferred" })
    String mode;

    @Param({ "128" })
    int dims;

    @Param({ "50000" })
    int numVectors;

    @Param({ "16" })
    int maxConn;

    @Param({ "100" })
    int beamWidth;

    private DeferredBuildService buildService;
    private float[][] vectors;
    private Path tempDir;

    private IndexWriter pendingWriter;
    private Directory pendingDir;

    @Setup(Level.Trial)
    public void setup() {
        buildService = new DeferredBuildService(Runtime.getRuntime().availableProcessors());

        vectors = new float[numVectors][dims];
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < numVectors; i++) {
            for (int d = 0; d < dims; d++) {
                vectors[i][d] = rng.nextFloat();
            }
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        if (buildService != null) {
            buildService.shutdown();
        }
    }

    @Setup(Level.Iteration)
    public void setupIteration() throws IOException {
        Path base = Path.of(System.getProperty("hnsw.bench.tmpdir", System.getProperty("java.io.tmpdir")));
        Files.createDirectories(base);
        tempDir = Files.createTempDirectory(base, "hnsw-bench");
    }

    @TearDown(Level.Iteration)
    public void tearDownIteration() throws Exception {
        if (pendingWriter != null) {
            pendingWriter.close();
            pendingWriter = null;
        }
        if (pendingDir != null) {
            pendingDir.close();
            pendingDir = null;
        }
        if (tempDir != null) {
            org.elasticsearch.core.IOUtils.rm(tempDir);
        }
    }

    private Codec createCodec() {
        boolean deferred = "deferred".equals(mode);
        return new Lucene912Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                if (deferred) {
                    return new ESHnswVectorsFormat(maxConn, beamWidth, buildService.vectorBuildService);
                }
                return new ESHnswVectorsFormat(maxConn, beamWidth, null);
            }
        };
    }

    @Benchmark
    public int addDocumentsOnly() throws Exception {
        Codec codec = createCodec();
        IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec).setRAMBufferSizeMB(4096);

        pendingDir = new MMapDirectory(tempDir.resolve("index"));
        pendingWriter = new IndexWriter(pendingDir, iwc);

        for (int i = 0; i < numVectors; i++) {
            Document doc = new Document();
            doc.add(new KnnFloatVectorField("vector", vectors[i], VectorSimilarityFunction.DOT_PRODUCT));
            pendingWriter.addDocument(doc);
        }
        return pendingWriter.getDocStats().numDocs;
    }

    @Benchmark
    public void indexAndFlush(Blackhole bh) throws Exception {
        Codec codec = createCodec();
        IndexWriterConfig iwc = new IndexWriterConfig().setCodec(codec).setRAMBufferSizeMB(512);

        try (Directory dir = new MMapDirectory(tempDir.resolve("index"))) {
            try (IndexWriter writer = new IndexWriter(dir, iwc)) {
                for (int i = 0; i < numVectors; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField("vector", vectors[i], VectorSimilarityFunction.DOT_PRODUCT));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                float[] query = vectors[0];
                for (LeafReaderContext ctx : reader.leaves()) {
                    TopDocs td = ctx.reader().searchNearestVectors("vector", query, 10, null, Integer.MAX_VALUE);
                    bh.consume(td);
                }
            }
        }
    }

    static class DeferredBuildService {
        final org.elasticsearch.index.engine.VectorBuildExecutorService vectorBuildService;
        private final org.elasticsearch.threadpool.ThreadPool threadPool;

        DeferredBuildService(int threads) {
            org.elasticsearch.common.settings.Settings settings = org.elasticsearch.common.settings.Settings.builder()
                .put("node.name", "bench")
                .build();
            this.threadPool = new org.elasticsearch.threadpool.ThreadPool(
                settings,
                org.elasticsearch.telemetry.metric.MeterRegistry.NOOP,
                new org.elasticsearch.threadpool.DefaultBuiltInExecutorBuilders()
            );
            this.vectorBuildService = new org.elasticsearch.index.engine.VectorBuildExecutorService(threadPool, threads);
        }

        void shutdown() {
            vectorBuildService.close();
            org.elasticsearch.threadpool.ThreadPool.terminate(threadPool, 10, java.util.concurrent.TimeUnit.SECONDS);
        }
    }
}
