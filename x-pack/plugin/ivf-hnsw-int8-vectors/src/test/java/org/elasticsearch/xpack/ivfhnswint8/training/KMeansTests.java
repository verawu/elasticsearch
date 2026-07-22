/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8.training;

import org.elasticsearch.test.ESTestCase;

import java.util.Random;

public class KMeansTests extends ESTestCase {

    public void testTrainTwoClusters() {
        float[][] vectors = new float[][] {
            { 0, 0 },
            { 1, 0 },
            { 0, 1 },
            { 10, 10 },
            { 11, 10 },
            { 10, 11 } };
        float[][] centroids = KMeans.train(vectors, 2, 100, new Random(42));
        assertEquals(2, centroids.length);

        int[] assignments = KMeans.assign(vectors, centroids);
        // First three should be in same cluster, last three in another
        assertEquals(assignments[0], assignments[1]);
        assertEquals(assignments[0], assignments[2]);
        assertEquals(assignments[3], assignments[4]);
        assertEquals(assignments[3], assignments[5]);
        assertNotEquals(assignments[0], assignments[3]);
    }

    public void testTrainSingleCluster() {
        float[][] vectors = new float[][] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        float[][] centroids = KMeans.train(vectors, 1, 10, new Random(0));
        assertEquals(1, centroids.length);
        // Centroid should be the mean
        assertEquals(4.0f, centroids[0][0], 0.01f);
        assertEquals(5.0f, centroids[0][1], 0.01f);
        assertEquals(6.0f, centroids[0][2], 0.01f);
    }

    public void testTrainKGreaterThanN() {
        float[][] vectors = new float[][] { { 1, 2 }, { 3, 4 } };
        float[][] centroids = KMeans.train(vectors, 10, 10, new Random(0));
        // k is clamped to n
        assertEquals(2, centroids.length);
    }

    public void testNearestCentroid() {
        float[][] centroids = new float[][] { { 0, 0 }, { 10, 10 }, { -10, -10 } };
        assertEquals(0, KMeans.nearestCentroid(new float[] { 1, 1 }, centroids));
        assertEquals(1, KMeans.nearestCentroid(new float[] { 9, 9 }, centroids));
        assertEquals(2, KMeans.nearestCentroid(new float[] { -8, -8 }, centroids));
    }

    public void testSquaredL2() {
        float[] a = { 1, 2, 3 };
        float[] b = { 4, 6, 3 };
        // (3^2 + 4^2 + 0^2) = 25
        assertEquals(25.0f, KMeans.squaredL2(a, b), 1e-6f);
    }

    public void testEmptyVectors() {
        expectThrows(IllegalArgumentException.class, () -> KMeans.train(new float[0][], 3, 10, new Random(0)));
    }

    public void testConvergenceWithLargerDataset() {
        Random random = new Random(123);
        int n = 1000;
        int dims = 8;
        int k = 10;
        float[][] vectors = new float[n][dims];
        for (int i = 0; i < n; i++) {
            int cluster = i % k;
            for (int d = 0; d < dims; d++) {
                vectors[i][d] = cluster * 100 + random.nextFloat() * 2 - 1;
            }
        }

        float[][] centroids = KMeans.train(vectors, k, 50, random);
        assertEquals(k, centroids.length);

        int[] assignments = KMeans.assign(vectors, centroids);
        // Vectors from the same planted cluster should mostly be assigned together
        int correct = 0;
        for (int i = 0; i < n; i++) {
            int plantedCluster = i % k;
            if (assignments[i] == assignments[plantedCluster]) {
                correct++;
            }
        }
        // Should get most right (> 80%) with well-separated clusters
        assertTrue("Expected high clustering accuracy, got " + correct + "/" + n, correct > n * 0.8);
    }
}
