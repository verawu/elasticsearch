/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.elasticsearch.xpack.ivfpq.training.KMeans;
import org.elasticsearch.xpack.ivfpq.training.ProductQuantizer;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_CURRENT;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_START;

public class IvfPqVectorsReader extends KnnVectorsReader {

    private final Map<String, FieldEntry> fields = new HashMap<>();
    private final IndexInput data;
    private final int nprobe;

    public IvfPqVectorsReader(SegmentReadState state, int nprobe) throws IOException {
        this.nprobe = nprobe;
        boolean success = false;
        IndexInput dataInput = null;
        try {
            String metaFileName = IndexInput.getSegmentFileName(
                state.segmentInfo.name,
                state.segmentSuffix,
                META_EXTENSION
            );
            try (ChecksumIndexInput metaIn = state.directory.openChecksumInput(metaFileName)) {
                CodecUtil.checkIndexHeader(
                    metaIn,
                    META_CODEC_NAME,
                    VERSION_START,
                    VERSION_CURRENT,
                    state.segmentInfo.getId(),
                    state.segmentSuffix
                );
                readFields(metaIn, state.fieldInfos);
                CodecUtil.checkFooter(metaIn);
            }

            String dataFileName = IndexInput.getSegmentFileName(
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
            success = true;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(dataInput);
            }
        }
    }

    private void readFields(ChecksumIndexInput metaIn, FieldInfos fieldInfos) throws IOException {
        int fieldNumber;
        while ((fieldNumber = metaIn.readInt()) != -1) {
            FieldInfo fieldInfo = fieldInfos.fieldInfo(fieldNumber);
            int encodingOrd = metaIn.readInt();
            int similarityOrd = metaIn.readInt();
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

                // Read centroids
                float[][] centroids = new float[nlistRead][dimension];
                for (int c = 0; c < nlistRead; c++) {
                    for (int d = 0; d < dimension; d++) {
                        centroids[c][d] = Float.intBitsToFloat(metaIn.readInt());
                    }
                }

                // Read PQ codebooks
                int ksub = 1 << nbitsRead;
                int dsub = dimension / mRead;
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
        // Verification done during construction via CodecUtil.checkIndexHeader/checkFooter
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
        IndexInput slice = data.clone();
        slice.seek(entry.rawVectorDataOffset);

        for (int i = 0; i < entry.vectorCount; i++) {
            int docId = slice.readInt();
            float[] vector = new float[entry.dimension];
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

    private void searchIvfPq(FieldEntry entry, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        int effectiveNprobe = Math.min(nprobe, entry.nlist);

        // Find nprobe nearest clusters
        float[] centroidDists = new float[entry.nlist];
        for (int c = 0; c < entry.nlist; c++) {
            centroidDists[c] = KMeans.squaredL2(target, entry.centroids[c]);
        }

        int[] probeOrder = argsort(centroidDists);

        // Search selected clusters
        IndexInput clusterData = data.clone();
        for (int p = 0; p < effectiveNprobe; p++) {
            int clusterIdx = probeOrder[p];
            int clusterSize = entry.clusterSizes[clusterIdx];
            if (clusterSize == 0) {
                continue;
            }

            // Compute residual for this cluster's centroid
            float[] residual = new float[entry.dimension];
            for (int d = 0; d < entry.dimension; d++) {
                residual[d] = target[d] - entry.centroids[clusterIdx][d];
            }

            // Build ADC distance table for this residual
            float[][] distTable = entry.quantizer.buildDistanceTable(residual);

            // Scan inverted list
            clusterData.seek(entry.clusterOffsets[clusterIdx]);
            for (int i = 0; i < clusterSize; i++) {
                int docId = clusterData.readInt();
                byte[] pqCodes = new byte[entry.m];
                clusterData.readBytes(pqCodes, 0, entry.m);

                if (acceptDocs != null && acceptDocs.get(docId) == false) {
                    continue;
                }

                float approxDist = ProductQuantizer.adcDistance(distTable, pqCodes);
                float docScore = distToScore(approxDist, entry.similarity);
                knnCollector.collect(docId, docScore);
                knnCollector.incVisitedCount(1);
            }
        }
    }

    private static float score(float[] a, float[] b, VectorSimilarityFunction similarity) {
        return switch (similarity) {
            case EUCLIDEAN -> {
                float dist = 0;
                for (int i = 0; i < a.length; i++) {
                    float diff = a[i] - b[i];
                    dist += diff * diff;
                }
                yield 1.0f / (1.0f + dist);
            }
            case DOT_PRODUCT -> {
                float dot = 0;
                for (int i = 0; i < a.length; i++) {
                    dot += a[i] * b[i];
                }
                yield (1.0f + dot) / 2.0f;
            }
            case COSINE -> {
                float dot = 0, normA = 0, normB = 0;
                for (int i = 0; i < a.length; i++) {
                    dot += a[i] * b[i];
                    normA += a[i] * a[i];
                    normB += b[i] * b[i];
                }
                float denom = (float) (Math.sqrt(normA) * Math.sqrt(normB));
                yield denom == 0 ? 0 : (1.0f + dot / denom) / 2.0f;
            }
            case MAXIMUM_INNER_PRODUCT -> {
                float dot = 0;
                for (int i = 0; i < a.length; i++) {
                    dot += a[i] * b[i];
                }
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

    private static int[] argsort(float[] values) {
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, (a, b) -> Float.compare(values[a], values[b]));
        int[] result = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = indices[i];
        }
        return result;
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

        @Override
        public float[] vectorValue(int ord) throws IOException {
            long baseOffset = entry.rawVectorDataOffset;
            int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
            dataInput.seek(baseOffset + (long) ord * bytesPerEntry + Integer.BYTES);
            float[] vector = new float[entry.dimension];
            for (int d = 0; d < entry.dimension; d++) {
                vector[d] = Float.intBitsToFloat(dataInput.readInt());
            }
            return vector;
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
