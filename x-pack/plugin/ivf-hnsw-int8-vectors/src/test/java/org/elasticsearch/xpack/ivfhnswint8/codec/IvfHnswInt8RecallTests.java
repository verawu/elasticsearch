/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8.codec;

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
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.test.ESTestCase;
import org.elasticsearch.xpack.ivfhnswint8.training.KMeans;

import java.io.IOException;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

public class IvfHnswInt8RecallTests extends ESTestCase {

    static {
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging();
    }

    private Codec getCodec(int nlist, int nprobe, int sqBits, int trainingThreshold) {
        return new Lucene912Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new IvfHnswInt8VectorsFormat(nlist, nprobe, sqBits, trainingThreshold, 20);
            }
        };
    }

    public void testRecallAt10() throws IOException {
        int dims = 32;
        int nVectors = 1000;
        int nQueries = 50;
        int k = 10;
        int nlist = 32;
        int nprobe = 16;

        Random random = new Random(42);
        float[][] vectors = new float[nVectors][dims];
        for (int i = 0; i < nVectors; i++) {
            vectors[i] = randomVector(dims, random);
        }

        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(nlist, nprobe, 7, 100));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                double totalRecall = 0;

                for (int q = 0; q < nQueries; q++) {
                    float[] query = randomVector(dims, random);

                    TopDocs approxResults = searcher.search(new KnnFloatVectorQuery("vec", query, k), k);

                    Set<Integer> exactTopK = bruteForceTopK(vectors, query, k);

                    int hits = 0;
                    for (ScoreDoc sd : approxResults.scoreDocs) {
                        if (exactTopK.contains(sd.doc)) {
                            hits++;
                        }
                    }
                    totalRecall += (double) hits / k;
                }

                double avgRecall = totalRecall / nQueries;
                assertTrue(
                    "Average recall@" + k + " should be > 0.7, was " + String.format("%.3f", avgRecall),
                    avgRecall > 0.7
                );
            }
        }
    }

    private Set<Integer> bruteForceTopK(float[][] vectors, float[] query, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Float.compare(
            Float.intBitsToFloat(b[1]), Float.intBitsToFloat(a[1])
        ));

        for (int i = 0; i < vectors.length; i++) {
            float dist = KMeans.squaredL2(query, vectors[i]);
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

    private static float[] randomVector(int dims, Random random) {
        float[] vector = new float[dims];
        for (int d = 0; d < dims; d++) {
            vector[d] = random.nextFloat() * 2 - 1;
        }
        return vector;
    }
}
