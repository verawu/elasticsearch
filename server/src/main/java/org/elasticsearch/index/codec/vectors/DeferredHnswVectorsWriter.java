/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.packed.DirectMonotonicWriter;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.engine.VectorBuildExecutorService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * A {@link KnnVectorsWriter} that defers HNSW graph construction from per-document indexing
 * to flush time. During {@link #addField}/{@code addValue}, vectors are stored only in the
 * flat format (O(1) per document). At {@link #flush} time, the HNSW graph is built from
 * scratch using {@link HnswGraphBuilder#build} on the VECTOR_BUILD thread pool.
 * <p>
 * The on-disk format is identical to {@link Lucene99HnswVectorsWriter}, so segments produced
 * by this writer are readable by {@link Lucene99HnswVectorsReader}.
 */
public final class DeferredHnswVectorsWriter extends KnnVectorsWriter {

    private static final long SHALLOW_RAM_BYTES_USED =
        RamUsageEstimator.shallowSizeOfInstance(DeferredHnswVectorsWriter.class);

    // Lucene99HnswVectorsFormat constants (package-private in Lucene, replicated here for format compatibility)
    static final String META_CODEC_NAME = "Lucene99HnswVectorsFormatMeta";
    static final String VECTOR_INDEX_CODEC_NAME = "Lucene99HnswVectorsFormatIndex";
    static final String META_EXTENSION = "vem";
    static final String VECTOR_INDEX_EXTENSION = "vex";
    static final int DIRECT_MONOTONIC_BLOCK_SHIFT = 16;

    private final SegmentWriteState segmentWriteState;
    private final IndexOutput meta;
    private final IndexOutput vectorIndex;
    private final int M;
    private final int beamWidth;
    private final FlatVectorsWriter flatVectorWriter;
    private final VectorBuildExecutorService buildService;
    private final String indexName;
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;

    private final List<DeferredFieldWriter<?>> fields = new ArrayList<>();
    private boolean finished;

    public DeferredHnswVectorsWriter(
        SegmentWriteState state,
        int M,
        int beamWidth,
        FlatVectorsWriter flatVectorWriter,
        VectorBuildExecutorService buildService,
        @Nullable String indexName,
        int numMergeWorkers,
        TaskExecutor mergeExec
    ) throws IOException {
        this.M = M;
        this.beamWidth = beamWidth;
        this.flatVectorWriter = flatVectorWriter;
        this.buildService = buildService;
        this.indexName = indexName != null ? indexName : "_unknown";
        this.numMergeWorkers = numMergeWorkers;
        this.mergeExec = mergeExec;
        this.segmentWriteState = state;

        String metaFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            META_EXTENSION
        );
        String indexDataFileName = IndexFileNames.segmentFileName(
            state.segmentInfo.name,
            state.segmentSuffix,
            VECTOR_INDEX_EXTENSION
        );

        boolean success = false;
        try {
            meta = state.directory.createOutput(metaFileName, state.context);
            vectorIndex = state.directory.createOutput(indexDataFileName, state.context);

            CodecUtil.writeIndexHeader(
                meta,
                META_CODEC_NAME,
                Lucene99HnswVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            CodecUtil.writeIndexHeader(
                vectorIndex,
                VECTOR_INDEX_CODEC_NAME,
                Lucene99HnswVectorsFormat.VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        FlatFieldVectorsWriter<?> flatFieldWriter = flatVectorWriter.addField(fieldInfo);
        DeferredFieldWriter<?> newField = new DeferredFieldWriter<>(fieldInfo, flatFieldWriter);
        fields.add(newField);
        return newField;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        flatVectorWriter.flush(maxDoc, sortMap);

        for (DeferredFieldWriter<?> field : fields) {
            int vectorCount = field.getDocsWithFieldSet().cardinality();
            if (vectorCount == 0) {
                writeEmptyField(field.fieldInfo);
                continue;
            }

            OnHeapHnswGraph graph = buildGraphAsync(field);

            if (sortMap == null) {
                writeField(field.fieldInfo, graph, vectorCount);
            } else {
                writeSortingField(field, graph, sortMap, vectorCount);
            }
        }
    }

    private OnHeapHnswGraph buildGraphAsync(DeferredFieldWriter<?> field) throws IOException {
        FlatVectorsScorer scorer = flatVectorWriter.getFlatVectorScorer();
        RandomVectorScorerSupplier scorerSupplier = field.createScorerSupplier(scorer);
        int vectorCount = field.getDocsWithFieldSet().cardinality();

        HnswGraphBuilder builder = HnswGraphBuilder.create(scorerSupplier, M, beamWidth, HnswGraphBuilder.randSeed);
        builder.setInfoStream(segmentWriteState.infoStream);

        long estimatedCostBytes = (long) vectorCount * field.fieldInfo.getVectorDimension() * 4;
        CompletableFuture<OnHeapHnswGraph> graphFuture = buildService.submitBuildTask(
            indexName,
            () -> builder.build(vectorCount),
            estimatedCostBytes
        );

        try {
            return graphFuture.join();
        } catch (Exception e) {
            Throwable cause = e.getCause() != null ? e.getCause() : e;
            throw new IOException("Failed to build HNSW graph asynchronously", cause);
        }
    }

    private void writeEmptyField(FieldInfo fieldInfo) throws IOException {
        long vectorIndexOffset = vectorIndex.getFilePointer();
        writeMeta(fieldInfo, vectorIndexOffset, 0, 0, null, null);
    }

    private void writeField(FieldInfo fieldInfo, OnHeapHnswGraph graph, int count) throws IOException {
        long vectorIndexOffset = vectorIndex.getFilePointer();
        int[][] graphLevelNodeOffsets = writeGraph(graph);
        long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;
        writeMeta(fieldInfo, vectorIndexOffset, vectorIndexLength, count, graph, graphLevelNodeOffsets);
    }

    private void writeSortingField(DeferredFieldWriter<?> fieldData, OnHeapHnswGraph graph, Sorter.DocMap sortMap, int count)
        throws IOException {
        int[] ordMap = new int[count];
        int[] oldOrdMap = new int[count];
        mapOldOrdToNewOrd(fieldData.getDocsWithFieldSet(), sortMap, oldOrdMap, ordMap, null);

        long vectorIndexOffset = vectorIndex.getFilePointer();
        int[][] graphLevelNodeOffsets = graph == null ? new int[0][] : new int[graph.numLevels()][];
        HnswGraph mockGraph = reconstructAndWriteGraph(graph, ordMap, oldOrdMap, graphLevelNodeOffsets);
        long vectorIndexLength = vectorIndex.getFilePointer() - vectorIndexOffset;
        writeMeta(fieldData.fieldInfo, vectorIndexOffset, vectorIndexLength, count, mockGraph, graphLevelNodeOffsets);
    }

    private HnswGraph reconstructAndWriteGraph(
        OnHeapHnswGraph graph,
        int[] newToOldMap,
        int[] oldToNewMap,
        int[][] levelNodeOffsets
    ) throws IOException {
        if (graph == null) return null;

        List<int[]> nodesByLevel = new ArrayList<>(graph.numLevels());
        nodesByLevel.add(null);

        int maxOrd = graph.size();
        HnswGraph.NodesIterator nodesOnLevel0 = graph.getNodesOnLevel(0);
        levelNodeOffsets[0] = new int[nodesOnLevel0.size()];
        while (nodesOnLevel0.hasNext()) {
            int node = nodesOnLevel0.nextInt();
            NeighborArray neighbors = graph.getNeighbors(0, newToOldMap[node]);
            long offset = vectorIndex.getFilePointer();
            reconstructAndWriteNeighbours(neighbors, oldToNewMap, maxOrd);
            levelNodeOffsets[0][node] = Math.toIntExact(vectorIndex.getFilePointer() - offset);
        }

        for (int level = 1; level < graph.numLevels(); level++) {
            HnswGraph.NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
            int[] newNodes = new int[nodesOnLevel.size()];
            for (int n = 0; nodesOnLevel.hasNext(); n++) {
                newNodes[n] = oldToNewMap[nodesOnLevel.nextInt()];
            }
            Arrays.sort(newNodes);
            nodesByLevel.add(newNodes);
            levelNodeOffsets[level] = new int[newNodes.length];
            int nodeOffsetIndex = 0;
            for (int node : newNodes) {
                NeighborArray neighbors = graph.getNeighbors(level, newToOldMap[node]);
                long offset = vectorIndex.getFilePointer();
                reconstructAndWriteNeighbours(neighbors, oldToNewMap, maxOrd);
                levelNodeOffsets[level][nodeOffsetIndex++] = Math.toIntExact(vectorIndex.getFilePointer() - offset);
            }
        }

        return new HnswGraph() {
            @Override
            public int nextNeighbor() {
                throw new UnsupportedOperationException("Not supported on a mock graph");
            }

            @Override
            public void seek(int level, int target) {
                throw new UnsupportedOperationException("Not supported on a mock graph");
            }

            @Override
            public int size() {
                return graph.size();
            }

            @Override
            public int numLevels() {
                return graph.numLevels();
            }

            @Override
            public int entryNode() {
                throw new UnsupportedOperationException("Not supported on a mock graph");
            }

            @Override
            public NodesIterator getNodesOnLevel(int level) {
                if (level == 0) {
                    return graph.getNodesOnLevel(0);
                } else {
                    return new HnswGraph.ArrayNodesIterator(nodesByLevel.get(level), nodesByLevel.get(level).length);
                }
            }
        };
    }

    private void reconstructAndWriteNeighbours(NeighborArray neighbors, int[] oldToNewMap, int maxOrd) throws IOException {
        int size = neighbors.size();
        vectorIndex.writeVInt(size);
        int[] nnodes = neighbors.nodes();
        for (int i = 0; i < size; i++) {
            nnodes[i] = oldToNewMap[nnodes[i]];
        }
        Arrays.sort(nnodes, 0, size);
        for (int i = size - 1; i > 0; --i) {
            assert nnodes[i] < maxOrd : "node too large: " + nnodes[i] + ">=" + maxOrd;
            nnodes[i] -= nnodes[i - 1];
        }
        for (int i = 0; i < size; i++) {
            vectorIndex.writeVInt(nnodes[i]);
        }
    }

    private int[][] writeGraph(OnHeapHnswGraph graph) throws IOException {
        if (graph == null) return new int[0][0];
        int countOnLevel0 = graph.size();
        int[][] offsets = new int[graph.numLevels()][];
        for (int level = 0; level < graph.numLevels(); level++) {
            int[] sortedNodes = HnswGraph.NodesIterator.getSortedNodes(graph.getNodesOnLevel(level));
            offsets[level] = new int[sortedNodes.length];
            int nodeOffsetId = 0;
            for (int node : sortedNodes) {
                NeighborArray neighbors = graph.getNeighbors(level, node);
                int size = neighbors.size();
                long offsetStart = vectorIndex.getFilePointer();
                vectorIndex.writeVInt(size);
                int[] nnodes = neighbors.nodes();
                Arrays.sort(nnodes, 0, size);
                for (int i = size - 1; i > 0; --i) {
                    assert nnodes[i] < countOnLevel0 : "node too large: " + nnodes[i] + ">=" + countOnLevel0;
                    nnodes[i] -= nnodes[i - 1];
                }
                for (int i = 0; i < size; i++) {
                    vectorIndex.writeVInt(nnodes[i]);
                }
                offsets[level][nodeOffsetId++] = Math.toIntExact(vectorIndex.getFilePointer() - offsetStart);
            }
        }
        return offsets;
    }

    private void writeMeta(
        FieldInfo field,
        long vectorIndexOffset,
        long vectorIndexLength,
        int count,
        HnswGraph graph,
        int[][] graphLevelNodeOffsets
    ) throws IOException {
        meta.writeInt(field.number);
        meta.writeInt(field.getVectorEncoding().ordinal());
        meta.writeInt(distFuncToOrd(field.getVectorSimilarityFunction()));
        meta.writeVLong(vectorIndexOffset);
        meta.writeVLong(vectorIndexLength);
        meta.writeVInt(field.getVectorDimension());
        meta.writeInt(count);
        meta.writeVInt(M);
        if (graph == null) {
            meta.writeVInt(0);
        } else {
            meta.writeVInt(graph.numLevels());
            long valueCount = 0;
            for (int level = 0; level < graph.numLevels(); level++) {
                HnswGraph.NodesIterator nodesOnLevel = graph.getNodesOnLevel(level);
                valueCount += nodesOnLevel.size();
                if (level > 0) {
                    int[] nol = new int[nodesOnLevel.size()];
                    int numberConsumed = nodesOnLevel.consume(nol);
                    Arrays.sort(nol);
                    assert numberConsumed == nodesOnLevel.size();
                    meta.writeVInt(nol.length);
                    for (int i = nodesOnLevel.size() - 1; i > 0; --i) {
                        nol[i] -= nol[i - 1];
                    }
                    for (int n : nol) {
                        assert n >= 0 : "delta encoding for nodes failed; expected nodes to be sorted";
                        meta.writeVInt(n);
                    }
                } else {
                    assert nodesOnLevel.size() == count : "Level 0 expects to have all nodes";
                }
            }
            long start = vectorIndex.getFilePointer();
            meta.writeLong(start);
            meta.writeVInt(DIRECT_MONOTONIC_BLOCK_SHIFT);
            final DirectMonotonicWriter memoryOffsetsWriter = DirectMonotonicWriter.getInstance(
                meta,
                vectorIndex,
                valueCount,
                DIRECT_MONOTONIC_BLOCK_SHIFT
            );
            long cumulativeOffsetSum = 0;
            for (int[] levelOffsets : graphLevelNodeOffsets) {
                for (int v : levelOffsets) {
                    memoryOffsetsWriter.add(cumulativeOffsetSum);
                    cumulativeOffsetSum += v;
                }
            }
            memoryOffsetsWriter.finish();
            meta.writeLong(vectorIndex.getFilePointer() - start);
        }
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        // Delegate merge to a standard Lucene99HnswVectorsWriter for full compatibility
        try (
            Lucene99HnswVectorsWriter mergeDelegate = new Lucene99HnswVectorsWriter(
                segmentWriteState,
                M,
                beamWidth,
                flatVectorWriter,
                numMergeWorkers,
                mergeExec
            )
        ) {
            mergeDelegate.mergeOneField(fieldInfo, mergeState);
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        flatVectorWriter.finish();
        if (meta != null) {
            meta.writeInt(-1);
            CodecUtil.writeFooter(meta);
        }
        if (vectorIndex != null) {
            CodecUtil.writeFooter(vectorIndex);
        }
    }

    @Override
    public long ramBytesUsed() {
        long total = SHALLOW_RAM_BYTES_USED;
        for (DeferredFieldWriter<?> field : fields) {
            total += field.ramBytesUsed();
        }
        return total;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorWriter, meta, vectorIndex);
    }

    static int distFuncToOrd(VectorSimilarityFunction func) {
        for (int i = 0; i < Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS.size(); i++) {
            if (Lucene99HnswVectorsReader.SIMILARITY_FUNCTIONS.get(i).equals(func)) {
                return i;
            }
        }
        throw new IllegalArgumentException("invalid distance function: " + func);
    }

    /**
     * Field writer that stores vectors in flat format only (no graph building during addValue).
     * The graph is built later at flush time.
     */
    private static class DeferredFieldWriter<T> extends KnnFieldVectorsWriter<T> {

        private static final long SHALLOW_SIZE =
            RamUsageEstimator.shallowSizeOfInstance(DeferredFieldWriter.class);

        final FieldInfo fieldInfo;
        private final FlatFieldVectorsWriter<T> flatFieldWriter;
        private int lastDocID = -1;
        private int node = 0;

        DeferredFieldWriter(FieldInfo fieldInfo, FlatFieldVectorsWriter<T> flatFieldWriter) {
            this.fieldInfo = fieldInfo;
            this.flatFieldWriter = flatFieldWriter;
        }

        @Override
        public void addValue(int docID, T vectorValue) throws IOException {
            if (docID == lastDocID) {
                throw new IllegalArgumentException(
                    "VectorValuesField \""
                        + fieldInfo.name
                        + "\" appears more than once in this document (only one value is allowed per field)"
                );
            }
            flatFieldWriter.addValue(docID, vectorValue);
            node++;
            lastDocID = docID;
        }

        @Override
        public T copyValue(T vectorValue) {
            throw new UnsupportedOperationException();
        }

        @Override
        public long ramBytesUsed() {
            return SHALLOW_SIZE + flatFieldWriter.ramBytesUsed();
        }

        DocsWithFieldSet getDocsWithFieldSet() {
            return flatFieldWriter.getDocsWithFieldSet();
        }

        @SuppressWarnings("unchecked")
        RandomVectorScorerSupplier createScorerSupplier(FlatVectorsScorer scorer) throws IOException {
            switch (fieldInfo.getVectorEncoding()) {
                case BYTE:
                    return scorer.getRandomVectorScorerSupplier(
                        fieldInfo.getVectorSimilarityFunction(),
                        RandomAccessVectorValues.fromBytes(
                            (List<byte[]>) flatFieldWriter.getVectors(),
                            fieldInfo.getVectorDimension()
                        )
                    );
                case FLOAT32:
                    return scorer.getRandomVectorScorerSupplier(
                        fieldInfo.getVectorSimilarityFunction(),
                        RandomAccessVectorValues.fromFloats(
                            (List<float[]>) flatFieldWriter.getVectors(),
                            fieldInfo.getVectorDimension()
                        )
                    );
                default:
                    throw new IllegalStateException("Unsupported vector encoding: " + fieldInfo.getVectorEncoding());
            }
        }
    }
}
