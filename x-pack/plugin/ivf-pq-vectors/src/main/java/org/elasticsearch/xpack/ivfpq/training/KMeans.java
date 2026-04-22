/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.training;

import java.util.Arrays;
import java.util.Random;

public class KMeans {

    public static float[][] train(float[][] vectors, int k, int maxIters, Random random) {
        if (vectors.length == 0) {
            throw new IllegalArgumentException("Cannot train k-means on empty vector set");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got " + k);
        }
        k = Math.min(k, vectors.length);
        int dims = vectors[0].length;

        float[][] centroids = initializePlusPlus(vectors, k, dims, random);

        int[] assignments = new int[vectors.length];
        for (int iter = 0; iter < maxIters; iter++) {
            boolean changed = assignAll(vectors, centroids, assignments);
            recomputeCentroids(vectors, centroids, assignments, k, dims, random);
            if (changed == false) {
                break;
            }
        }
        return centroids;
    }

    public static int[] assign(float[][] vectors, float[][] centroids) {
        int[] assignments = new int[vectors.length];
        assignAll(vectors, centroids, assignments);
        return assignments;
    }

    public static int nearestCentroid(float[] vector, float[][] centroids) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < centroids.length; i++) {
            float dist = squaredL2(vector, centroids[i]);
            if (dist < bestDist) {
                bestDist = dist;
                best = i;
            }
        }
        return best;
    }

    private static float[][] initializePlusPlus(float[][] vectors, int k, int dims, Random random) {
        float[][] centroids = new float[k][dims];
        int firstIdx = random.nextInt(vectors.length);
        System.arraycopy(vectors[firstIdx], 0, centroids[0], 0, dims);

        float[] minDists = new float[vectors.length];
        Arrays.fill(minDists, Float.MAX_VALUE);

        for (int c = 1; c < k; c++) {
            double totalDist = 0;
            for (int i = 0; i < vectors.length; i++) {
                float dist = squaredL2(vectors[i], centroids[c - 1]);
                if (dist < minDists[i]) {
                    minDists[i] = dist;
                }
                totalDist += minDists[i];
            }

            double threshold = random.nextDouble() * totalDist;
            double cumulative = 0;
            int selected = vectors.length - 1;
            for (int i = 0; i < vectors.length; i++) {
                cumulative += minDists[i];
                if (cumulative >= threshold) {
                    selected = i;
                    break;
                }
            }
            System.arraycopy(vectors[selected], 0, centroids[c], 0, dims);
        }
        return centroids;
    }

    private static boolean assignAll(float[][] vectors, float[][] centroids, int[] assignments) {
        boolean changed = false;
        for (int i = 0; i < vectors.length; i++) {
            int nearest = nearestCentroid(vectors[i], centroids);
            if (nearest != assignments[i]) {
                assignments[i] = nearest;
                changed = true;
            }
        }
        return changed;
    }

    private static void recomputeCentroids(
        float[][] vectors,
        float[][] centroids,
        int[] assignments,
        int k,
        int dims,
        Random random
    ) {
        int[] counts = new int[k];
        for (float[] centroid : centroids) {
            Arrays.fill(centroid, 0);
        }

        for (int i = 0; i < vectors.length; i++) {
            int cluster = assignments[i];
            counts[cluster]++;
            for (int d = 0; d < dims; d++) {
                centroids[cluster][d] += vectors[i][d];
            }
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < dims; d++) {
                    centroids[c][d] /= counts[c];
                }
            } else {
                // Reinitialize empty cluster from a random vector
                int randomIdx = random.nextInt(vectors.length);
                System.arraycopy(vectors[randomIdx], 0, centroids[c], 0, dims);
            }
        }
    }

    static float squaredL2(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}
