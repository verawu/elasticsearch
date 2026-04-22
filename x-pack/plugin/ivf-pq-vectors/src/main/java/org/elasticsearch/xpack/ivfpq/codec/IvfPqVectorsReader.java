/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.xpack.ivfpq.training.KMeans;
import org.elasticsearch.xpack.ivfpq.training.ProductQuantizer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_CURRENT;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_START;

public class IvfPqVectorsReader extends KnnVectorsReader {

    private static final ThreadLocal<Integer> NPROBE_OVERRIDE = new ThreadLocal<>();

    public static void setNprobeOverride(int nprobe) {
        NPROBE_OVERRIDE.set(nprobe);
    }

    public static void clearNprobeOverride() {
        NPROBE_OVERRIDE.remove();
    }

    private final Map<String, FieldEntry> fields;
    private final IndexInput data;
    private final int nprobe;

    public IvfPqVectorsReader(SegmentReadState state, int nprobe) throws IOException {
        this.nprobe = nprobe;
        boolean success = false;
        IndexInput dataInput = null;
        Map<String, FieldEntry> tempFields = new HashMap<>();
        try {
            String metaFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                META_EXTENSION
            );
            try (ChecksumIndexInput metaIn = state.directory.openChecksumInput(metaFileName, state.context)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(metaIn, state.fieldInfos, tempFields);
                CodecUtil.checkFooter(metaIn);
            }

            String dataFileName = IndexFileNames.segmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                DATA_EXTENSION
            );
            dataInput = state.directory.openInput(dataFileName, IOContext.DEFAULT);
            CodecUtil.checkIndexHeader(
                dataInput,
                DATA_CODEC_NAME,
                VERSION_START,
                VERSION_CURRENT,
                state.segmentInfo.getId(),
                state.segmentSuffix
            );
            this.data = dataInput;
            this.fields = Collections.unmodifiableMap(tempFields);
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(dataInput);
            }
        }
    }

    private void readFields(ChecksumIndexInput metaIn, FieldInfos fieldInfos, Map<String, FieldEntry> fields) throws IOException {
        int fieldNumber;
        while ((fieldNumber = metaIn.readInt()) != -1) {
            FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber);
            if (fieldInfo == null) {
                throw new CorruptIndexException("Invalid field number: " + fieldNumber, metaIn);
            }
            int encodingOrd = metaIn.readInt();
            int similarityOrd = metaIn.readInt();
            if (encodingOrd < 0 || encodingOrd >= VectorEncoding.values().length) {
                throw new CorruptIndexException("Invalid vector encoding ordinal: " + encodingOrd, metaIn);
            }
            if (similarityOrd < 0 || similarityOrd >= VectorSimilarityFunction.values().length) {
                throw new CorruptIndexException("Invalid similarity function ordinal: " + similarityOrd, metaIn);
            }
            VectorEncoding encoding = VectorEncoding.values()[encodingOrd];
            VectorSimilarityFunction similarity = VectorSimilarityFunction.values()[similarityOrd];
            int dimension = metaIn.readVInt();
            boolean isFlat = metaIn.readByte() == 1;
            int vectorCount = metaIn.readVInt();

            if (isFlat) {
                long dataOffset = metaIn.readVLong();
                long dataLength = metaIn.readVLong();
                fields.put(
                    fieldInfo.name,
                    new FieldEntry(
                        fieldInfo.name,
                        similarity,
                        encoding,
                        dimension,
                        vectorCount,
                        true,
                        0,
                        0,
                        0,
                        null,
                        null,
                        null,
                        null,
                        dataOffset,
                        dataLength,
                        0,
                        0
                    )
                );
            } else {
                int nlistRead = metaIn.readVInt();
                int mRead = metaIn.readVInt();
                int nbitsRead = metaIn.readVInt();
                int ksub = metaIn.readVInt();
                int dsub = dimension / mRead;

                // Read centroids
                float[][] centroids = new float[nlistRead][dimension];
                for (int c = 0; c < nlistRead; c++) {
                    for (int d = 0; d < dimension; d++) {
                        centroids[c][d] = Float.intBitsToFloat(metaIn.readInt());
                    }
                }
                float[][][] codebooks = new float[mRead][ksub][dsub];
                for (int sub = 0; sub < mRead; sub++) {
                    for (int code = 0; code < ksub; code++) {
                        for (int d = 0; d < dsub; d++) {
                            codebooks[sub][code][d] = Float.intBitsToFloat(metaIn.readInt());
                        }
                    }
                }

                ProductQuantizer pq = new ProductQuantizer(mRead, ksub, dsub, codebooks);

                // Read cluster metadata
                long[] clusterOffsets = new long[nlistRead];
                int[] clusterSizes = new int[nlistRead];
                for (int c = 0; c < nlistRead; c++) {
                    clusterOffsets[c] = metaIn.readVLong();
                    clusterSizes[c] = metaIn.readVInt();
                }

                // Raw vector data
                long rawVectorDataOffset = metaIn.readVLong();
                long rawVectorDataLength = metaIn.readVLong();

                fields.put(
                    fieldInfo.name,
                    new FieldEntry(
                        fieldInfo.name,
                        similarity,
                        encoding,
                        dimension,
                        vectorCount,
                        false,
                        nlistRead,
                        mRead,
                        nbitsRead,
                        centroids,
                        pq,
                        clusterOffsets,
                        clusterSizes,
                        rawVectorDataOffset,
                        rawVectorDataLength,
                        0,
                        0
                    )
                );
            }
        }
    }

    @Override
    public void checkIntegrity() throws IOException {
        CodecUtil.checksumEntireFile(data);
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        FieldEntry entry = fields.get(field);
        if (entry == null) {
            return null;
        }
        return new IvfPqFloatVectorValues(entry, data.clone());
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return null;
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        FieldEntry entry = fields.get(field);
        if (entry == null || entry.vectorCount == 0) {
            return;
        }

        if (entry.isFlat) {
            searchFlat(entry, target, knnCollector, acceptDocs);
        } else {
            searchIvfPq(entry, target, knnCollector, acceptDocs);
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        throw new UnsupportedOperationException("IVF_PQ format only supports float vectors");
    }

    private void searchFlat(FieldEntry entry, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        try (IndexInput slice = data.clone()) {
            slice.seek(entry.rawVectorDataOffset);
            float[] vector = new float[entry.dimension];

            for (int i = 0; i < entry.vectorCount; i++) {
                int docId = slice.readInt();
                for (int d = 0; d < entry.dimension; d++) {
                    vector[d] = Float.intBitsToFloat(slice.readInt());
                }

                if (acceptDocs != null && acceptDocs.get(docId) == false) {
                    continue;
                }

                float score = score(target, vector, entry.similarity);
                knnCollector.collect(docId, score);
                knnCollector.incVisitedCount(1);
            }
        }
    }

    private void searchIvfPq(FieldEntry entry, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        Integer overrideNprobe = NPROBE_OVERRIDE.get();
        int baseNprobe = overrideNprobe != null ? overrideNprobe : nprobe;
        int effectiveNprobe = Math.min(baseNprobe, entry.nlist);

        // Find nprobe nearest clusters
        float[] centroidDists = new float[entry.nlist];
        for (int c = 0; c < entry.nlist; c++) {
            centroidDists[c] = KMeans.squaredL2(target, entry.centroids[c]);
        }

        int[] probeOrder = topKIndices(centroidDists, effectiveNprobe);

        // Phase 1: collect ADC candidates
        List<int[]> candidates = new ArrayList<>();
        try (IndexInput clusterData = data.clone()) {
            byte[] pqCodes = new byte[entry.m];
            for (int p = 0; p < effectiveNprobe; p++) {
                int clusterIdx = probeOrder[p];
                int clusterSize = entry.clusterSizes[clusterIdx];
                if (clusterSize == 0) {
                    continue;
                }

                float[] residual = new float[entry.dimension];
                for (int d = 0; d < entry.dimension; d++) {
                    residual[d] = target[d] - entry.centroids[clusterIdx][d];
                }

                float[][] distTable = entry.quantizer.buildDistanceTable(residual);

                clusterData.seek(entry.clusterOffsets[clusterIdx]);
                for (int i = 0; i < clusterSize; i++) {
                    int docId = clusterData.readInt();
                    clusterData.readBytes(pqCodes, 0, entry.m);

                    if (acceptDocs != null && acceptDocs.get(docId) == false) {
                        continue;
                    }

                    float approxDist = ProductQuantizer.adcDistance(distTable, pqCodes);
                    candidates.add(new int[] { docId, Float.floatToIntBits(approxDist) });
                    knnCollector.incVisitedCount(1);
                }
            }
        }

        if (candidates.isEmpty()) {
            return;
        }

        // Sort candidates by approximate distance (ascending)
        candidates.sort((a, b) -> Float.compare(Float.intBitsToFloat(a[1]), Float.intBitsToFloat(b[1])));

        // Phase 2: rerank top candidates with exact vectors
        int rerankCount = Math.min(candidates.size(), knnCollector.k());
        int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
        try (IndexInput rawData = data.clone()) {
            float[] vector = new float[entry.dimension];
            for (int i = 0; i < rerankCount; i++) {
                int docId = candidates.get(i)[0];
                int ord = findOrdinalByDocId(rawData, entry, docId, bytesPerEntry);
                if (ord >= 0) {
                    rawData.seek(entry.rawVectorDataOffset + (long) ord * bytesPerEntry + Integer.BYTES);
                    for (int d = 0; d < entry.dimension; d++) {
                        vector[d] = Float.intBitsToFloat(rawData.readInt());
                    }
                    float exactScore = score(target, vector, entry.similarity);
                    knnCollector.collect(docId, exactScore);
                } else {
                    float docScore = distToScore(Float.intBitsToFloat(candidates.get(i)[1]), entry.similarity);
                    knnCollector.collect(docId, docScore);
                }
            }
        }
    }

    private static int findOrdinalByDocId(IndexInput rawData, FieldEntry entry, int targetDocId, int bytesPerEntry)
        throws IOException {
        int lo = 0, hi = entry.vectorCount - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            rawData.seek(entry.rawVectorDataOffset + (long) mid * bytesPerEntry);
            int docId = rawData.readInt();
            if (docId == targetDocId) {
                return mid;
            } else if (docId < targetDocId) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    static float score(float[] a, float[] b, VectorSimilarityFunction similarity) {
        return switch (similarity) {
            case EUCLIDEAN -> {
                float dist = VectorUtil.squareDistance(a, b);
                yield 1.0f / (1.0f + dist);
            }
            case DOT_PRODUCT -> {
                float dot = VectorUtil.dotProduct(a, b);
                yield (1.0f + dot) / 2.0f;
            }
            case COSINE -> {
                float dot = VectorUtil.dotProduct(a, b);
                float normA = VectorUtil.dotProduct(a, a);
                float normB = VectorUtil.dotProduct(b, b);
                float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
                yield denom == 0 ? 0 : (1.0f + dot / denom) / 2.0f;
            }
            case MAXIMUM_INNER_PRODUCT -> {
                float dot = VectorUtil.dotProduct(a, b);
                yield dot >= 0 ? dot + 1.0f : 1.0f / (1.0f - dot);
            }
        };
    }

    private static float distToScore(float squaredL2Dist, VectorSimilarityFunction similarity) {
        return switch (similarity) {
            case EUCLIDEAN -> 1.0f / (1.0f + squaredL2Dist);
            case DOT_PRODUCT, COSINE -> {
                // ADC computes squared L2 on residuals; approximate as negative distance
                // For non-euclidean similarities, this is an approximation
                yield 1.0f / (1.0f + squaredL2Dist);
            }
            case MAXIMUM_INNER_PRODUCT -> 1.0f / (1.0f + squaredL2Dist);
        };
    }

    private static int[] topKIndices(float[] values, int k) {
        k = Math.min(k, values.length);
        int[] indices = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            indices[i] = i;
        }
        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < values.length; j++) {
                if (Float.compare(values[indices[j]], values[indices[minIdx]]) < 0) {
                    minIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[minIdx];
            indices[minIdx] = tmp;
        }
        return indices;
    }

    @Override
    public long ramBytesUsed() {
        long bytes = 0;
        for (FieldEntry entry : fields.values()) {
            if (entry.centroids != null) {
                bytes += (long) entry.nlist * entry.dimension * Float.BYTES;
            }
            if (entry.quantizer != null) {
                bytes += (long) entry.m * entry.quantizer.getKsub() * entry.quantizer.getDsub() * Float.BYTES;
            }
            if (entry.clusterOffsets != null) {
                bytes += (long) entry.clusterOffsets.length * Long.BYTES;
            }
            if (entry.clusterSizes != null) {
                bytes += (long) entry.clusterSizes.length * Integer.BYTES;
            }
        }
        return bytes;
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(data);
    }

    record FieldEntry(
        String fieldName,
        VectorSimilarityFunction similarity,
        VectorEncoding encoding,
        int dimension,
        int vectorCount,
        boolean isFlat,
        int nlist,
        int m,
        int nbits,
        float[][] centroids,
        ProductQuantizer quantizer,
        long[] clusterOffsets,
        int[] clusterSizes,
        long rawVectorDataOffset,
        long rawVectorDataLength,
        long flatDataOffset,
        long flatDataLength
    ) {}

    static class IvfPqFloatVectorValues extends FloatVectorValues {
        private final FieldEntry entry;
        private final IndexInput dataInput;
        private int ord = -1;
        private int docId = -1;
        private final float[] scratch;

        IvfPqFloatVectorValues(FieldEntry entry, IndexInput dataInput) {
            this.entry = entry;
            this.dataInput = dataInput;
            this.scratch = new float[entry.dimension];
        }

        @Override
        public int dimension() {
            return entry.dimension;
        }

        @Override
        public int size() {
            return entry.vectorCount;
        }

        @Override
        public float[] vectorValue() throws IOException {
            long baseOffset = entry.rawVectorDataOffset;
            int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
            dataInput.seek(baseOffset + (long) ord * bytesPerEntry + Integer.BYTES);
            for (int d = 0; d < entry.dimension; d++) {
                scratch[d] = Float.intBitsToFloat(dataInput.readInt());
            }
            return scratch;
        }

        @Override
        public int docID() {
            return docId;
        }

        @Override
        public int nextDoc() throws IOException {
            if (ord + 1 >= entry.vectorCount) {
                docId = DocIdSetIterator.NO_MORE_DOCS;
                return docId;
            }
            ord++;
            long baseOffset = entry.rawVectorDataOffset;
            int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
            dataInput.seek(baseOffset + (long) ord * bytesPerEntry);
            docId = dataInput.readInt();
            return docId;
        }

        @Override
        public int advance(int target) throws IOException {
            while (nextDoc() < target) {
                // advance
            }
            return docId;
        }

        public float[] vectorValue(int ord) throws IOException {
            long baseOffset = entry.rawVectorDataOffset;
            int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
            dataInput.seek(baseOffset + (long) ord * bytesPerEntry + Integer.BYTES);
            for (int d = 0; d < entry.dimension; d++) {
                scratch[d] = Float.intBitsToFloat(dataInput.readInt());
            }
            return scratch;
        }

        public void close() throws IOException {
            dataInput.close();
        }

        @Override
        public VectorScorer scorer(float[] target) throws IOException {
            IvfPqFloatVectorValues copy = new IvfPqFloatVectorValues(entry, dataInput.clone());
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    float[] vec = copy.vectorValue();
                    return IvfPqVectorsReader.score(target, vec, entry.similarity);
                }

                @Override
                public DocIdSetIterator iterator() {
                    return copy;
                }
            };
        }
    }
}
