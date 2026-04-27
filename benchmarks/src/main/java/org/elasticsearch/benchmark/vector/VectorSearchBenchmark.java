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
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.ES813FlatVectorFormat;
import org.elasticsearch.index.codec.vectors.ES814HnswScalarQuantizedVectorsFormat;
import org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat;
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

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@Fork(value = 1)
@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Thread)
public class VectorSearchBenchmark {

    static {
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging();
    }

    private static final int NUM_QUERIES = 100;
    private static final String FIELD = "vec";

    @Param({ "flat", "int8_hnsw", "ivfpq" })
    String format;

    @Param({ "128", "768" })
    int dims;

    @Param({ "10000", "100000" })
    int numVectors;

    @Param({ "10" })
    int k;

    @Param({ "0" })
    int numCandidates;

    @Param({ "0" })
    int nprobeParam;

    @Param({ "uniform", "clustered" })
    String dataDistribution;

    private Directory dir;
    private DirectoryReader reader;
    private IndexSearcher searcher;
    private float[][] queries;
    private int queryIndex;
    private Path tempDir;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        Random random = new Random(42);
        float[][] vectors;
        if ("clustered".equals(dataDistribution)) {
            vectors = generateClusteredVectors(numVectors, dims, 64, 0.05f, random);
            queries = generateClusteredQueries(NUM_QUERIES, dims, vectors, 0.05f, random);
        } else if ("wide_clustered".equals(dataDistribution)) {
            vectors = generateClusteredVectors(numVectors, dims, 256, 0.2f, random);
            queries = generateClusteredQueries(NUM_QUERIES, dims, vectors, 0.2f, random);
        } else {
            vectors = new float[numVectors][dims];
            for (int i = 0; i < numVectors; i++) {
                vectors[i] = randomVector(dims, random);
            }
            queries = new float[NUM_QUERIES][dims];
            for (int i = 0; i < NUM_QUERIES; i++) {
                queries[i] = randomVector(dims, random);
            }
        }

        tempDir = Files.createTempDirectory("vector-bench-");
        dir = MMapDirectory.open(tempDir);

        long startTime = System.nanoTime();
        IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(buildCodec());
        config.setRAMBufferSizeMB(256);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (float[] vector : vectors) {
                Document doc = new Document();
                doc.add(new KnnFloatVectorField(FIELD, vector, VectorSimilarityFunction.EUCLIDEAN));
                writer.addDocument(doc);
            }
            writer.commit();
        }
        long indexTimeMs = (System.nanoTime() - startTime) / 1_000_000;

        long indexSizeBytes = directorySize(tempDir);

        reader = DirectoryReader.open(dir);
        searcher = new IndexSearcher(reader);

        int efSearch = numCandidates > 0 ? numCandidates : k;
        double totalRecall = 0;
        for (int q = 0; q < NUM_QUERIES; q++) {
            Set<Integer> groundTruth = bruteForceTopK(vectors, queries[q], k);
            TopDocs results = searcher.search(new KnnFloatVectorQuery(FIELD, queries[q], efSearch), k);
            int hits = 0;
            for (ScoreDoc sd : results.scoreDocs) {
                if (groundTruth.contains(sd.doc)) {
                    hits++;
                }
            }
            totalRecall += (double) hits / k;
        }
        double avgRecall = totalRecall / NUM_QUERIES;

        System.out.printf(
            "[%s] data=%s, dims=%d, n=%d, numCandidates=%d: indexTime=%dms, indexSize=%.1fMB, recall@%d=%.3f%n",
            format,
            dataDistribution,
            dims,
            numVectors,
            efSearch,
            indexTimeMs,
            indexSizeBytes / (1024.0 * 1024.0),
            k,
            avgRecall
        );
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        if (reader != null) reader.close();
        if (dir != null) dir.close();
    }

    @Benchmark
    public float queryLatency() throws IOException {
        float[] q = queries[queryIndex++ % NUM_QUERIES];
        int efSearch = numCandidates > 0 ? numCandidates : k;
        TopDocs results = searcher.search(new KnnFloatVectorQuery(FIELD, q, efSearch), k);
        if (results.scoreDocs.length == 0) {
            return 0f;
        }
        return results.scoreDocs[0].score;
    }

    private Codec buildCodec() {
        int nlist = Math.max(4, (int) Math.sqrt(numVectors));
        KnnVectorsFormat vectorsFormat = switch (format) {
            case "flat" -> new ES813FlatVectorFormat();
            case "int8_hnsw" -> new ES814HnswScalarQuantizedVectorsFormat(16, 100, null, 7, false);
            case "ivfpq" -> {
                int nprobe = nprobeParam > 0 ? nprobeParam : Math.min(nlist, 32);
                yield new IvfPqVectorsFormat(nlist, nprobe, 7, 100, 25);
            }
            default -> throw new IllegalArgumentException("Unknown format: " + format);
        };
        return new Lucene912Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return vectorsFormat;
            }
        };
    }

    private static float[][] generateClusteredVectors(int n, int dims, int numClusters, float stddev, Random random) {
        float[][] centroids = new float[numClusters][dims];
        for (int c = 0; c < numClusters; c++) {
            centroids[c] = randomVector(dims, random);
        }
        float[][] vectors = new float[n][dims];
        for (int i = 0; i < n; i++) {
            int cluster = random.nextInt(numClusters);
            vectors[i] = new float[dims];
            for (int d = 0; d < dims; d++) {
                vectors[i][d] = centroids[cluster][d] + (float) (random.nextGaussian() * stddev);
            }
        }
        return vectors;
    }

    private static float[][] generateClusteredQueries(int numQueries, int dims, float[][] vectors, float stddev, Random random) {
        float[][] queries = new float[numQueries][dims];
        for (int q = 0; q < numQueries; q++) {
            float[] base = vectors[random.nextInt(vectors.length)];
            queries[q] = new float[dims];
            for (int d = 0; d < dims; d++) {
                queries[q][d] = base[d] + (float) (random.nextGaussian() * stddev * 0.5);
            }
        }
        return queries;
    }

    private static Set<Integer> bruteForceTopK(float[][] vectors, float[] query, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Float.compare(
            Float.intBitsToFloat(b[1]), Float.intBitsToFloat(a[1])
        ));
        for (int i = 0; i < vectors.length; i++) {
            float dist = squaredL2(query, vectors[i]);
            int distBits = Float.floatToIntBits(dist);
            if (pq.size() < k) {
                pq.add(new int[] { i, distBits });
            } else if (dist < Float.intBitsToFloat(pq.peek()[1])) {
                pq.poll();
                pq.add(new int[] { i, distBits });
            }
        }
        Set<Integer> topK = new HashSet<>();
        for (int[] entry : pq) {
            topK.add(entry[0]);
        }
        return topK;
    }

    private static float squaredL2(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private static float[] randomVector(int dims, Random random) {
        float[] v = new float[dims];
        for (int d = 0; d < dims; d++) {
            v[d] = random.nextFloat() * 2 - 1;
        }
        return v;
    }

    private static long directorySize(Path dir) throws IOException {
        long size = 0;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isRegularFile(entry)) {
                    size += Files.size(entry);
                }
            }
        }
        return size;
    }
}
