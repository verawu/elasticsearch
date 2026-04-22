/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.elasticsearch.xpack.ivfpq.training.KMeans;
import org.elasticsearch.xpack.ivfpq.training.ProductQuantizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_CURRENT;

public class IvfPqVectorsWriter extends KnnVectorsWriter {

    private final SegmentWriteState state;
    private final IndexOutput meta;
    private final IndexOutput data;
    private final int nlist;
    private final int m;
    private final int nbits;
    private final int trainingThreshold;
    private final int kmeansIters;
    private final List<FieldWriter> fields = new ArrayList<>();
    private boolean finished = false;

    public IvfPqVectorsWriter(SegmentWriteState state, int nlist, int m, int nbits, int trainingThreshold, int kmeansIters)
        throws IOException {
        this.state = state;
        this.nlist = nlist;
        this.m = m;
        this.nbits = nbits;
        this.trainingThreshold = trainingThreshold;
        this.kmeansIters = kmeansIters;

        boolean success = false;
        try {
            String metaFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                META_EXTENSION
            );
            meta = state.directory.createOutput(metaFileName, state.context);
            CodecUtil.writeIndexHeader(meta, META_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);

            String dataFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                DATA_EXTENSION
            );
            data = state.directory.createOutput(dataFileName, state.context);
            CodecUtil.writeIndexHeader(data, DATA_CODEC_NAME, VERSION_CURRENT, state.segmentInfo.getId(), state.segmentSuffix);

            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(this);
            }
        }
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        FieldWriter writer = new FieldWriter(fieldInfo);
        fields.add(writer);
        return writer;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        for (FieldWriter fieldWriter : fields) {
            if (fieldWriter.vectors.isEmpty()) {
                continue;
            }
            float[][] vectors = fieldWriter.vectors.toArray(new float[0][]);
            int[] docIds = fieldWriter.docIds.stream().mapToInt(Integer::intValue).toArray();

            if (sortMap != null) {
                int[] newDocIds = new int[docIds.length];
                for (int i = 0; i < docIds.length; i++) {
                    newDocIds[i] = sortMap.oldToNew(docIds[i]);
                }
                Integer[] indices = new Integer[vectors.length];
                for (int i = 0; i < indices.length; i++) {
                    indices[i] = i;
                }
                Arrays.sort(indices, (a, b) -> Integer.compare(newDocIds[a], newDocIds[b]));
                float[][] sortedVectors = new float[vectors.length][];
                int[] sortedDocIds = new int[docIds.length];
                for (int i = 0; i < indices.length; i++) {
                    sortedVectors[i] = vectors[indices[i]];
                    sortedDocIds[i] = newDocIds[indices[i]];
                }
                vectors = sortedVectors;
                docIds = sortedDocIds;
            }

            writeField(fieldWriter.fieldInfo, vectors, docIds);
        }
    }

    private void writeField(FieldInfo fieldInfo, float[][] vectors, int[] docIds) throws IOException {
        int dims = fieldInfo.getVectorDimension();
        VectorSimilarityFunction similarity = fieldInfo.getVectorSimilarityFunction();
        boolean isFlat = vectors.length < trainingThreshold;

        meta.writeInt(fieldInfo.number);
        meta.writeInt(fieldInfo.getVectorEncoding().ordinal());
        meta.writeInt(similarity.ordinal());
        meta.writeVInt(dims);
        meta.writeByte((byte) (isFlat ? 1 : 0));
        meta.writeVInt(vectors.length);

        if (isFlat) {
            long dataOffset = data.getFilePointer();
            for (int i = 0; i < vectors.length; i++) {
                data.writeInt(docIds[i]);
                for (int d = 0; d < dims; d++) {
                    data.writeInt(Float.floatToIntBits(vectors[i][d]));
                }
            }
            long dataLength = data.getFilePointer() - dataOffset;
            meta.writeVLong(dataOffset);
            meta.writeVLong(dataLength);
        } else {
            writeIvfPqField(fieldInfo, vectors, docIds, dims, similarity);
        }
    }

    private void writeIvfPqField(FieldInfo fieldInfo, float[][] vectors, int[] docIds, int dims, VectorSimilarityFunction similarity)
        throws IOException {
        int effectiveNlist = Math.min(nlist, vectors.length);
        int effectiveM = m;
        Random random = new Random(42);

        // Train IVF centroids
        float[][] centroids = KMeans.train(vectors, effectiveNlist, kmeansIters, random);
        effectiveNlist = centroids.length;

        // Assign vectors to clusters
        int[] assignments = KMeans.assign(vectors, centroids);

        // Compute residuals
        float[][] residuals = new float[vectors.length][dims];
        for (int i = 0; i < vectors.length; i++) {
            for (int d = 0; d < dims; d++) {
                residuals[i][d] = vectors[i][d] - centroids[assignments[i]][d];
            }
        }

        // Train PQ codebooks on residuals
        ProductQuantizer pq = ProductQuantizer.train(residuals, dims, effectiveM, nbits, kmeansIters, random);

        // PQ-encode all residuals
        byte[][] pqCodes = new byte[vectors.length][];
        for (int i = 0; i < vectors.length; i++) {
            pqCodes[i] = pq.encode(residuals[i]);
        }

        // Build per-cluster inverted lists
        @SuppressWarnings({ "unchecked", "rawtypes" })
        List<Integer>[] clusterMembers = new List[effectiveNlist];
        for (int c = 0; c < effectiveNlist; c++) {
            clusterMembers[c] = new ArrayList<>();
        }
        for (int i = 0; i < vectors.length; i++) {
            clusterMembers[assignments[i]].add(i);
        }

        // Write inverted lists to data file
        long[] clusterOffsets = new long[effectiveNlist];
        int[] clusterSizes = new int[effectiveNlist];
        for (int c = 0; c < effectiveNlist; c++) {
            clusterOffsets[c] = data.getFilePointer();
            clusterSizes[c] = clusterMembers[c].size();
            for (int idx : clusterMembers[c]) {
                data.writeInt(docIds[idx]);
                data.writeBytes(pqCodes[idx], 0, pqCodes[idx].length);
            }
        }

        // Write raw vectors for reranking/merge
        long rawVectorDataOffset = data.getFilePointer();
        for (int i = 0; i < vectors.length; i++) {
            data.writeInt(docIds[i]);
            for (int d = 0; d < dims; d++) {
                data.writeInt(Float.floatToIntBits(vectors[i][d]));
            }
        }
        long rawVectorDataLength = data.getFilePointer() - rawVectorDataOffset;

        // Write metadata
        meta.writeVInt(effectiveNlist);
        meta.writeVInt(effectiveM);
        meta.writeVInt(nbits);
        meta.writeVInt(pq.getKsub());

        // Centroids
        for (int c = 0; c < effectiveNlist; c++) {
            for (int d = 0; d < dims; d++) {
                meta.writeInt(Float.floatToIntBits(centroids[c][d]));
            }
        }

        // PQ Codebooks: m * ksub * dsub floats
        float[][][] codebooks = pq.getCodebooks();
        int ksub = pq.getKsub();
        int dsub = pq.getDsub();
        for (int sub = 0; sub < effectiveM; sub++) {
            for (int code = 0; code < ksub; code++) {
                for (int d = 0; d < dsub; d++) {
                    meta.writeInt(Float.floatToIntBits(codebooks[sub][code][d]));
                }
            }
        }

        // Cluster metadata
        for (int c = 0; c < effectiveNlist; c++) {
            meta.writeVLong(clusterOffsets[c]);
            meta.writeVInt(clusterSizes[c]);
        }

        // Raw vector data offset
        meta.writeVLong(rawVectorDataOffset);
        meta.writeVLong(rawVectorDataLength);
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        FloatVectorValues mergedValues = KnnVectorsWriter.MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);

        List<float[]> vectorsList = new ArrayList<>();
        List<Integer> docIdsList = new ArrayList<>();
        while (mergedValues.nextDoc() != FloatVectorValues.NO_MORE_DOCS) {
            vectorsList.add(Arrays.copyOf(mergedValues.vectorValue(), mergedValues.vectorValue().length));
            docIdsList.add(mergedValues.docID());
        }

        if (vectorsList.isEmpty()) {
            return;
        }

        float[][] vectors = vectorsList.toArray(new float[0][]);
        int[] docIds = docIdsList.stream().mapToInt(Integer::intValue).toArray();
        writeField(fieldInfo, vectors, docIds);
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        finished = true;
        meta.writeInt(-1); // end of fields sentinel
        CodecUtil.writeFooter(meta);
        CodecUtil.writeFooter(data);
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(meta, data);
    }

    @Override
    public long ramBytesUsed() {
        long bytes = 0;
        for (FieldWriter fieldWriter : fields) {
            bytes += (long) fieldWriter.vectors.size() * fieldWriter.fieldInfo.getVectorDimension() * Float.BYTES;
        }
        return bytes;
    }

    static class FieldWriter extends KnnFieldVectorsWriter<float[]> {
        final FieldInfo fieldInfo;
        final List<float[]> vectors = new ArrayList<>();
        final List<Integer> docIds = new ArrayList<>();

        FieldWriter(FieldInfo fieldInfo) {
            this.fieldInfo = fieldInfo;
        }

        @Override
        public void addValue(int docID, float[] vectorValue) throws IOException {
            vectors.add(Arrays.copyOf(vectorValue, vectorValue.length));
            docIds.add(docID);
        }

        @Override
        public float[] copyValue(float[] vectorValue) {
            return Arrays.copyOf(vectorValue, vectorValue.length);
        }

        @Override
        public long ramBytesUsed() {
            return (long) vectors.size() * fieldInfo.getVectorDimension() * Float.BYTES;
        }
    }
}
