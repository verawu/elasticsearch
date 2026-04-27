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
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.quantization.ScalarQuantizer;
import org.elasticsearch.xpack.ivfpq.training.KMeans;

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

    private static final String IVFPQ_COMPONENT = "IvfPqVectors";

    private final SegmentWriteState state;
    private final IndexOutput meta;
    private final IndexOutput data;
    private final int nlist;
    private final byte sqBits;
    private final int trainingThreshold;
    private final int kmeansIters;
    private final List<FieldWriter> fields = new ArrayList<>();
    private boolean finished = false;

    public IvfPqVectorsWriter(SegmentWriteState state, int nlist, int sqBits, int trainingThreshold, int kmeansIters)
        throws IOException {
        this.state = state;
        this.nlist = nlist;
        this.sqBits = (byte) sqBits;
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

        if (state.infoStream.isEnabled(IVFPQ_COMPONENT)) {
            state.infoStream.message(
                IVFPQ_COMPONENT,
                "field [" + fieldInfo.name + "] vectors=" + vectors.length + " mode=" + (isFlat ? "flat" : "ivf-sq")
            );
        }

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
            writeIvfSqField(fieldInfo, vectors, docIds, dims, similarity);
        }
    }

    private void writeIvfSqField(FieldInfo fieldInfo, float[][] vectors, int[] docIds, int dims, VectorSimilarityFunction similarity)
        throws IOException {
        int effectiveNlist = Math.min(nlist, vectors.length);
        long seed = Arrays.hashCode(state.segmentInfo.getId()) ^ fieldInfo.name.hashCode();
        Random random = new Random(seed);

        float[][] trainingVectors = vectors;
        int maxTrainingSize = 10 * trainingThreshold;
        if (vectors.length > maxTrainingSize) {
            trainingVectors = subsample(vectors, maxTrainingSize, random);
        }

        long trainStart = System.nanoTime();

        // Train IVF centroids
        float[][] centroids = KMeans.train(trainingVectors, effectiveNlist, kmeansIters, random);
        effectiveNlist = centroids.length;

        // Assign vectors to clusters
        int[] assignments = KMeans.assign(vectors, centroids);

        // Train global scalar quantizer
        ScalarQuantizer sq = ScalarQuantizer.fromVectorsAutoInterval(
            new InMemoryFloatVectorValues(vectors, dims),
            similarity,
            vectors.length,
            sqBits
        );

        long trainDurationMs = (System.nanoTime() - trainStart) / 1_000_000;
        if (state.infoStream.isEnabled(IVFPQ_COMPONENT)) {
            state.infoStream.message(
                IVFPQ_COMPONENT,
                "field [" + fieldInfo.name + "] trained IVF-SQ: nlist=" + effectiveNlist + " sqBits=" + sqBits + " vectors="
                    + vectors.length + " trainingSize=" + trainingVectors.length + " duration=" + trainDurationMs + "ms"
            );
        }

        // Scalar quantize all vectors
        byte[][] sqCodes = new byte[vectors.length][dims];
        float[] corrections = new float[vectors.length];
        for (int i = 0; i < vectors.length; i++) {
            corrections[i] = sq.quantize(vectors[i], sqCodes[i], similarity);
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

        // Sort each cluster's members by distance to centroid (nearest first)
        // and compute per-cluster radii for early termination
        float[] clusterRadii = new float[effectiveNlist];
        for (int c = 0; c < effectiveNlist; c++) {
            List<Integer> members = clusterMembers[c];
            if (members.isEmpty()) continue;
            float[] dists = new float[members.size()];
            for (int j = 0; j < members.size(); j++) {
                dists[j] = KMeans.squaredL2(vectors[members.get(j)], centroids[c]);
            }
            Integer[] order = new Integer[members.size()];
            for (int j = 0; j < order.length; j++) order[j] = j;
            Arrays.sort(order, (a, b) -> Float.compare(dists[a], dists[b]));
            List<Integer> sorted = new ArrayList<>(members.size());
            for (int idx : order) sorted.add(members.get(idx));
            clusterMembers[c] = sorted;
            clusterRadii[c] = dists[order[order.length - 1]];
        }

        // Write inverted lists to data file
        long[] clusterOffsets = new long[effectiveNlist];
        int[] clusterSizes = new int[effectiveNlist];
        for (int c = 0; c < effectiveNlist; c++) {
            clusterOffsets[c] = data.getFilePointer();
            clusterSizes[c] = clusterMembers[c].size();
            for (int idx : clusterMembers[c]) {
                data.writeInt(docIds[idx]);
                data.writeBytes(sqCodes[idx], 0, dims);
                data.writeInt(Float.floatToIntBits(corrections[idx]));
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
        meta.writeVInt(sqBits);
        meta.writeInt(Float.floatToIntBits(sq.getLowerQuantile()));
        meta.writeInt(Float.floatToIntBits(sq.getUpperQuantile()));

        // Centroids
        for (int c = 0; c < effectiveNlist; c++) {
            for (int d = 0; d < dims; d++) {
                meta.writeInt(Float.floatToIntBits(centroids[c][d]));
            }
        }

        // Cluster metadata
        for (int c = 0; c < effectiveNlist; c++) {
            meta.writeVLong(clusterOffsets[c]);
            meta.writeVInt(clusterSizes[c]);
        }

        // Cluster radii (max squared distance from centroid to any member)
        for (int c = 0; c < effectiveNlist; c++) {
            meta.writeInt(Float.floatToIntBits(clusterRadii[c]));
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
            float[] v = mergedValues.vectorValue();
            vectorsList.add(Arrays.copyOf(v, v.length));
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
        if (finished == false) {
            IOUtils.closeWhileHandlingException(meta, data);
            return;
        }
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

    private static float[][] subsample(float[][] vectors, int sampleSize, Random random) {
        float[][] sampled = new float[sampleSize][];
        boolean[] selected = new boolean[vectors.length];
        for (int i = 0; i < sampleSize; i++) {
            int idx;
            do {
                idx = random.nextInt(vectors.length);
            } while (selected[idx]);
            selected[idx] = true;
            sampled[i] = vectors[idx];
        }
        return sampled;
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

    static class InMemoryFloatVectorValues extends FloatVectorValues {
        private final float[][] vectors;
        private final int dims;
        private int ord = -1;
        private int doc = -1;

        InMemoryFloatVectorValues(float[][] vectors, int dims) {
            this.vectors = vectors;
            this.dims = dims;
        }

        @Override
        public int dimension() {
            return dims;
        }

        @Override
        public int size() {
            return vectors.length;
        }

        @Override
        public float[] vectorValue() {
            return vectors[ord];
        }

        @Override
        public int docID() {
            return doc;
        }

        @Override
        public int nextDoc() {
            if (ord + 1 >= vectors.length) {
                doc = NO_MORE_DOCS;
                return doc;
            }
            ord++;
            doc = ord;
            return doc;
        }

        @Override
        public int advance(int target) {
            while (nextDoc() < target) {
                // advance
            }
            return doc;
        }

        @Override
        public VectorScorer scorer(float[] target) {
            throw new UnsupportedOperationException();
        }
    }
}
