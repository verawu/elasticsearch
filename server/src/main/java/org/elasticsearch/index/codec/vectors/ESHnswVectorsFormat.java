/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.engine.VectorBuildExecutorService;

import java.io.IOException;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.MAX_DIMS_COUNT;

/**
 * ES wrapper around Lucene's HNSW vectors format that supports deferred graph building
 * via {@link DeferredHnswVectorsWriter} when a {@link VectorBuildExecutorService} is provided.
 */
public final class ESHnswVectorsFormat extends KnnVectorsFormat {

    static final String NAME = "ESHnswVectorsFormat";

    private final int maxConn;
    private final int beamWidth;
    private final FlatVectorsFormat flatVectorsFormat;
    private final VectorBuildExecutorService buildService;

    public ESHnswVectorsFormat(int maxConn, int beamWidth, @Nullable VectorBuildExecutorService buildService) {
        super(NAME);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.flatVectorsFormat = new Lucene99FlatVectorsFormat(DefaultFlatVectorScorer.INSTANCE);
        this.buildService = buildService;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        if (buildService != null) {
            return new DeferredHnswVectorsWriter(state, maxConn, beamWidth, flatVectorsFormat.fieldsWriter(state), buildService, 1, null);
        }
        return new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, flatVectorsFormat.fieldsWriter(state), 1, null);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return MAX_DIMS_COUNT;
    }

    @Override
    public String toString() {
        return "ESHnswVectorsFormat(name="
            + NAME
            + ", maxConn="
            + maxConn
            + ", beamWidth="
            + beamWidth
            + ", flatVectorFormat="
            + flatVectorsFormat
            + ")";
    }
}
