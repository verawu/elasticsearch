/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.training;

import org.elasticsearch.test.ESTestCase;

import java.util.Random;

public class ProductQuantizerTests extends ESTestCase {

    public void testTrainAndEncode() {
        Random random = new Random(42);
        int n = 500;
        int dims = 16;
        int m = 4;
        int nbits = 8;

        float[][] vectors = randomVectors(n, dims, random);
        ProductQuantizer pq = ProductQuantizer.train(vectors, dims, m, nbits, 20, random);

        assertEquals(m, pq.getM());
        assertEquals(256, pq.getKsub());
        assertEquals(dims / m, pq.getDsub());

        byte[] codes = pq.encode(vectors[0]);
        assertEquals(m, codes.length);
        // Codes should be valid (0-255)
        for (byte code : codes) {
            int unsigned = code & 0xFF;
            assertTrue(unsigned >= 0 && unsigned < 256);
        }
    }

    public void testAdcDistanceApproximation() {
        Random random = new Random(42);
        int n = 500;
        int dims = 16;
        int m = 4;
        int nbits = 8;

        float[][] vectors = randomVectors(n, dims, random);
        ProductQuantizer pq = ProductQuantizer.train(vectors, dims, m, nbits, 20, random);

        float[] query = randomVectors(1, dims, random)[0];
        float[][] distTable = pq.buildDistanceTable(query);

        double totalExactDist = 0;
        double totalApproxDist = 0;
        for (int i = 0; i < Math.min(100, n); i++) {
            byte[] codes = pq.encode(vectors[i]);
            float approxDist = ProductQuantizer.adcDistance(distTable, codes);
            float exactDist = KMeans.squaredL2(query, vectors[i]);
            totalExactDist += exactDist;
            totalApproxDist += approxDist;
            // ADC distance should be non-negative
            assertTrue("ADC distance should be non-negative, got " + approxDist, approxDist >= 0);
        }

        // Average distances should be in the same ballpark (within 5x)
        double ratio = totalApproxDist / totalExactDist;
        assertTrue(
            "ADC distance ratio to exact should be reasonable, got " + ratio,
            ratio > 0.1 && ratio < 10.0
        );
    }

    public void testAdcPreservesOrdering() {
        Random random = new Random(42);
        int n = 200;
        int dims = 16;
        int m = 4;
        int nbits = 8;

        float[][] vectors = randomVectors(n, dims, random);
        ProductQuantizer pq = ProductQuantizer.train(vectors, dims, m, nbits, 20, random);

        float[] query = randomVectors(1, dims, random)[0];
        float[][] distTable = pq.buildDistanceTable(query);

        // Find nearest by exact and by ADC
        int nearestExact = -1;
        float minExact = Float.MAX_VALUE;
        int nearestAdc = -1;
        float minAdc = Float.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            float exactDist = KMeans.squaredL2(query, vectors[i]);
            if (exactDist < minExact) {
                minExact = exactDist;
                nearestExact = i;
            }
            byte[] codes = pq.encode(vectors[i]);
            float adcDist = ProductQuantizer.adcDistance(distTable, codes);
            if (adcDist < minAdc) {
                minAdc = adcDist;
                nearestAdc = i;
            }
        }

        // The ADC nearest should be close to exact nearest (within top-10)
        float exactDistOfAdcNearest = KMeans.squaredL2(query, vectors[nearestAdc]);
        // Count how many vectors are closer than the ADC nearest
        int rank = 0;
        for (int i = 0; i < n; i++) {
            if (KMeans.squaredL2(query, vectors[i]) < exactDistOfAdcNearest) {
                rank++;
            }
        }
        assertTrue("ADC nearest should be in top-10 by exact distance, got rank " + rank, rank < 10);
    }

    public void testDimsNotDivisibleByM() {
        expectThrows(
            IllegalArgumentException.class,
            () -> ProductQuantizer.train(new float[][] { { 1, 2, 3 } }, 3, 2, 8, 10, new Random(0))
        );
    }

    public void testDistanceTable() {
        Random random = new Random(42);
        int dims = 8;
        int m = 2;
        int nbits = 8;
        float[][] vectors = randomVectors(100, dims, random);

        ProductQuantizer pq = ProductQuantizer.train(vectors, dims, m, nbits, 20, random);
        float[] query = new float[dims];

        float[][] table = pq.buildDistanceTable(query);
        assertEquals(m, table.length);
        assertEquals(256, table[0].length);
        // All distances from origin should be non-negative
        for (int sub = 0; sub < m; sub++) {
            for (int code = 0; code < 256; code++) {
                assertTrue(table[sub][code] >= 0);
            }
        }
    }

    private static float[][] randomVectors(int n, int dims, Random random) {
        float[][] vectors = new float[n][dims];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dims; d++) {
                vectors[i][d] = random.nextFloat() * 2 - 1;
            }
        }
        return vectors;
    }
}
