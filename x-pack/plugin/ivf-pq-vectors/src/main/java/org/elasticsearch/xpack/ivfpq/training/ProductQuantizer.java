/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.training;

import java.util.Random;

public class ProductQuantizer {

    private final int m;
    private final int ksub;
    private final int dsub;
    private final float[][][] codebooks; // [m][ksub][dsub]

    public ProductQuantizer(int m, int ksub, int dsub, float[][][] codebooks) {
        this.m = m;
        this.ksub = ksub;
        this.dsub = dsub;
        this.codebooks = codebooks;
    }

    public static ProductQuantizer train(float[][] vectors, int dims, int m, int nbits, int maxIters, Random random) {
        if (dims % m != 0) {
            throw new IllegalArgumentException("dims (" + dims + ") must be divisible by m (" + m + ")");
        }
        int dsub = dims / m;
        int ksub = 1 << nbits;

        float[][][] codebooks = new float[m][][];
        for (int sub = 0; sub < m; sub++) {
            float[][] subVectors = extractSubspace(vectors, sub, dsub);
            codebooks[sub] = KMeans.train(subVectors, ksub, maxIters, random);
        }
        int actualKsub = codebooks[0].length;
        return new ProductQuantizer(m, actualKsub, dsub, codebooks);
    }

    public byte[] encode(float[] vector) {
        byte[] codes = new byte[m];
        for (int sub = 0; sub < m; sub++) {
            codes[sub] = (byte) KMeans.nearestCentroid(vector, sub * dsub, dsub, codebooks[sub]);
        }
        return codes;
    }

    public float[][] buildDistanceTable(float[] query) {
        float[][] table = new float[m][ksub];
        for (int sub = 0; sub < m; sub++) {
            int offset = sub * dsub;
            for (int code = 0; code < ksub; code++) {
                table[sub][code] = KMeans.squaredL2(query, offset, codebooks[sub][code], 0, dsub);
            }
        }
        return table;
    }

    public static float adcDistance(float[][] distTable, byte[] codes) {
        float dist = 0;
        for (int sub = 0; sub < codes.length; sub++) {
            dist += distTable[sub][codes[sub] & 0xFF];
        }
        return dist;
    }

    public int getM() {
        return m;
    }

    public int getKsub() {
        return ksub;
    }

    public int getDsub() {
        return dsub;
    }

    public float[][][] getCodebooks() {
        float[][][] copy = new float[m][][];
        for (int sub = 0; sub < m; sub++) {
            copy[sub] = new float[ksub][];
            for (int code = 0; code < ksub; code++) {
                copy[sub][code] = codebooks[sub][code].clone();
            }
        }
        return copy;
    }

    private static float[][] extractSubspace(float[][] vectors, int subIndex, int dsub) {
        float[][] sub = new float[vectors.length][dsub];
        int offset = subIndex * dsub;
        for (int i = 0; i < vectors.length; i++) {
            System.arraycopy(vectors[i], offset, sub[i], 0, dsub);
        }
        return sub;
    }
}
