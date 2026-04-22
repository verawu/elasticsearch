/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.codec;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

import java.io.IOException;

public class IvfPqVectorsFormat extends KnnVectorsFormat {

    public static final String NAME = "IvfPqVectorsFormat";
    static final String META_CODEC_NAME = "IvfPqVectorsFormatMeta";
    static final String DATA_CODEC_NAME = "IvfPqVectorsFormatData";
    static final String META_EXTENSION = "ivfm";
    static final String DATA_EXTENSION = "ivfd";
    static final int VERSION_START = 0;
    static final int VERSION_CURRENT = VERSION_START;

    private final int nlist;
    private final int nprobe;
    private final int m;
    private final int nbits;
    private final int trainingThreshold;
    private final int kmeansIters;

    public IvfPqVectorsFormat() {
        this(256, 16, 8, 8, 1000, 20);
    }

    public IvfPqVectorsFormat(int nlist, int nprobe, int m, int nbits, int trainingThreshold, int kmeansIters) {
        super(NAME);
        this.nlist = nlist;
        this.nprobe = nprobe;
        this.m = m;
        this.nbits = nbits;
        this.trainingThreshold = trainingThreshold;
        this.kmeansIters = kmeansIters;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new IvfPqVectorsWriter(state, nlist, m, nbits, trainingThreshold, kmeansIters);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new IvfPqVectorsReader(state, nprobe);
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return 4096;
    }

    @Override
    public String toString() {
        return NAME + "(nlist=" + nlist + ", nprobe=" + nprobe + ", m=" + m + ", nbits=" + nbits + ")";
    }
}
