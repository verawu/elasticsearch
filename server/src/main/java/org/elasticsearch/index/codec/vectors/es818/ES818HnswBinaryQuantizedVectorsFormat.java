/*
 * @notice
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications copyright (C) 2024 Elasticsearch B.V.
 */
package org.elasticsearch.index.codec.vectors.es818;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.codec.vectors.DeferredHnswVectorsWriter;
import org.elasticsearch.index.engine.VectorBuildExecutorService;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN;
import static org.elasticsearch.index.mapper.vectors.DenseVectorFieldMapper.MAX_DIMS_COUNT;

/**
 * Copied from Lucene, replace with Lucene's implementation sometime after Lucene 10
 */
public class ES818HnswBinaryQuantizedVectorsFormat extends KnnVectorsFormat {

    public static final String NAME = "ES818HnswBinaryQuantizedVectorsFormat";

    /**
     * Controls how many of the nearest neighbor candidates are connected to the new node. Defaults to
     * {@link Lucene99HnswVectorsFormat#DEFAULT_MAX_CONN}. See {@link HnswGraph} for more details.
     */
    private final int maxConn;

    /**
     * The number of candidate neighbors to track while searching the graph for each newly inserted
     * node. Defaults to {@link Lucene99HnswVectorsFormat#DEFAULT_BEAM_WIDTH}. See {@link HnswGraph}
     * for details.
     */
    private final int beamWidth;

    /** The format for storing, reading, merging vectors on disk */
    private static final FlatVectorsFormat flatVectorsFormat = new ES818BinaryQuantizedVectorsFormat();

    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;

    @Nullable
    private final VectorBuildExecutorService buildService;

    /** Constructs a format using default graph construction parameters */
    public ES818HnswBinaryQuantizedVectorsFormat() {
        this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null, null);
    }

    public ES818HnswBinaryQuantizedVectorsFormat(int maxConn, int beamWidth) {
        this(maxConn, beamWidth, DEFAULT_NUM_MERGE_WORKER, null, null);
    }

    public ES818HnswBinaryQuantizedVectorsFormat(int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
        this(maxConn, beamWidth, numMergeWorkers, mergeExec, null);
    }

    public ES818HnswBinaryQuantizedVectorsFormat(
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        @Nullable VectorBuildExecutorService buildService
    ) {
        super(NAME);
        if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
            throw new IllegalArgumentException(
                "maxConn must be positive and less than or equal to " + MAXIMUM_MAX_CONN + "; maxConn=" + maxConn
            );
        }
        if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
            throw new IllegalArgumentException(
                "beamWidth must be positive and less than or equal to " + MAXIMUM_BEAM_WIDTH + "; beamWidth=" + beamWidth
            );
        }
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        if (numMergeWorkers == 1 && mergeExec != null) {
            throw new IllegalArgumentException("No executor service is needed as we'll use single thread to merge");
        }
        this.numMergeWorkers = numMergeWorkers;
        if (mergeExec != null) {
            this.mergeExec = new TaskExecutor(mergeExec);
        } else {
            this.mergeExec = null;
        }
        this.buildService = buildService;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        if (buildService != null) {
            return new DeferredHnswVectorsWriter(
                state,
                maxConn,
                beamWidth,
                flatVectorsFormat.fieldsWriter(state),
                buildService,
                numMergeWorkers,
                mergeExec
            );
        }
        return new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, flatVectorsFormat.fieldsWriter(state), numMergeWorkers, mergeExec);
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
        return "ES818HnswBinaryQuantizedVectorsFormat(name=ES818HnswBinaryQuantizedVectorsFormat, maxConn="
            + maxConn
            + ", beamWidth="
            + beamWidth
            + ", flatVectorFormat="
            + flatVectorsFormat
            + ")";
    }
}
