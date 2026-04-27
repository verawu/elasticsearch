/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.ScalarQuantizedVectorSimilarity;
import org.apache.lucene.util.quantization.ScalarQuantizer;
import org.elasticsearch.xpack.ivfpq.training.KMeans;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.DATA_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_CODEC_NAME;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.META_EXTENSION;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_CURRENT;
import static org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat.VERSION_START;

public class IvfPqVectorsReader extends KnnVectorsReader {

    private static final int HNSW_COARSE_MIN_NLIST = 64;
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
    private final boolean rerank;

    public IvfPqVectorsReader(SegmentReadState state, int nprobe, boolean rerank) throws IOException {
        this.nprobe = nprobe;
        this.rerank = rerank;

        Map<String, FieldEntry> tempFields = new HashMap<>();
        String metaFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, META_EXTENSION);
        try (ChecksumIndexInput metaIn = state.directory.openChecksumInput(metaFileName, IOContext.READONCE)) {
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

        IndexInput dataInput = null;
        boolean success = false;
        try {
            String dataFileName = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, DATA_EXTENSION);
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
                        (byte) 0,
                        null,
                        null,
                        null,
                        null,
                        null,
                        dataOffset,
                        dataLength,
                        0,
                        0,
                        null,
                        null
                    )
                );
            } else {
                int nlistRead = metaIn.readVInt();
                int sqBitsRead = metaIn.readVInt();
                float lowerQuantile = Float.intBitsToFloat(metaIn.readInt());
                float upperQuantile = Float.intBitsToFloat(metaIn.readInt());
                ScalarQuantizer sq = new ScalarQuantizer(lowerQuantile, upperQuantile, (byte) sqBitsRead);

                // Read centroids
                float[][] centroids = new float[nlistRead][dimension];
                for (int c = 0; c < nlistRead; c++) {
                    for (int d = 0; d < dimension; d++) {
                        centroids[c][d] = Float.intBitsToFloat(metaIn.readInt());
                    }
                }

                // Read cluster metadata
                long[] clusterOffsets = new long[nlistRead];
                int[] clusterSizes = new int[nlistRead];
                for (int c = 0; c < nlistRead; c++) {
                    clusterOffsets[c] = metaIn.readVLong();
                    clusterSizes[c] = metaIn.readVInt();
                }

                // Read cluster radii (max squared distance from centroid to any member)
                float[] clusterRadii = new float[nlistRead];
                for (int c = 0; c < nlistRead; c++) {
                    clusterRadii[c] = Float.intBitsToFloat(metaIn.readInt());
                }

                // Raw vector data
                long rawVectorDataOffset = metaIn.readVLong();
                long rawVectorDataLength = metaIn.readVLong();

                // Build HNSW graph over centroids for fast coarse quantizer lookup
                OnHeapHnswGraph centroidGraph = null;
                RandomAccessVectorValues.Floats centroidValues = null;
                if (nlistRead >= HNSW_COARSE_MIN_NLIST) {
                    centroidValues = RandomAccessVectorValues.fromFloats(Arrays.asList(centroids), dimension);
                    RandomVectorScorerSupplier scorerSupplier = DefaultFlatVectorScorer.INSTANCE
                        .getRandomVectorScorerSupplier(VectorSimilarityFunction.EUCLIDEAN, centroidValues);
                    centroidGraph = HnswGraphBuilder.create(scorerSupplier, 16, 100, 42L, nlistRead)
                        .build(nlistRead);
                }

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
                        (byte) sqBitsRead,
                        centroids,
                        sq,
                        clusterOffsets,
                        clusterSizes,
                        clusterRadii,
                        rawVectorDataOffset,
                        rawVectorDataLength,
                        0,
                        0,
                        centroidGraph,
                        centroidValues
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
            searchIvfSq(entry, target, knnCollector, acceptDocs);
        }
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        throw new UnsupportedOperationException("IVF format only supports float vectors");
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

    private void searchIvfSq(FieldEntry entry, float[] target, KnnCollector knnCollector, Bits acceptDocs) throws IOException {
        Integer overrideNprobe = NPROBE_OVERRIDE.get();
        int baseNprobe = overrideNprobe != null ? overrideNprobe : nprobe;
        int effectiveNprobe = Math.min(baseNprobe, entry.nlist);

        // Find nprobe nearest clusters
        int[] probeOrder;
        if (entry.centroidGraph != null) {
            RandomVectorScorer queryScorer = DefaultFlatVectorScorer.INSTANCE
                .getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, entry.centroidValues, target);
            TopDocs nearest = HnswGraphSearcher.search(
                queryScorer, effectiveNprobe, entry.centroidGraph, null, Integer.MAX_VALUE
            ).topDocs();
            probeOrder = new int[nearest.scoreDocs.length];
            for (int i = 0; i < nearest.scoreDocs.length; i++) {
                probeOrder[i] = nearest.scoreDocs[i].doc;
            }
        } else {
            float[] centroidDists = new float[entry.nlist];
            for (int c = 0; c < entry.nlist; c++) {
                centroidDists[c] = KMeans.squaredL2(target, entry.centroids[c]);
            }
            probeOrder = topKIndices(centroidDists, effectiveNprobe);
        }

        // Quantize query once
        byte[] queryBytes = new byte[entry.dimension];
        float queryCorrection = entry.scalarQuantizer.quantize(target, queryBytes, entry.similarity);
        ScalarQuantizedVectorSimilarity sqScorer = ScalarQuantizedVectorSimilarity.fromVectorSimilarity(
            entry.similarity,
            entry.scalarQuantizer.getConstantMultiplier(),
            entry.sqBits
        );

        // Phase 1: collect SQ candidates with early termination
        int candidateCapacity = Math.min(knnCollector.k() * 4, entry.vectorCount);
        PriorityQueue<int[]> topCandidates = new PriorityQueue<>(
            (a, b) -> Float.compare(Float.intBitsToFloat(a[1]), Float.intBitsToFloat(b[1]))
        );
        float kthBestScore = Float.NEGATIVE_INFINITY;

        try (IndexInput clusterData = data.clone()) {
            byte[] storedBytes = new byte[entry.dimension];
            for (int p = 0; p < probeOrder.length; p++) {
                int clusterIdx = probeOrder[p];
                int clusterSize = entry.clusterSizes[clusterIdx];
                if (clusterSize == 0) continue;

                // Inter-cluster early termination (EUCLIDEAN only)
                if (topCandidates.size() >= knnCollector.k()
                    && entry.clusterRadii != null
                    && entry.similarity == VectorSimilarityFunction.EUCLIDEAN) {
                    float centroidDist = KMeans.squaredL2(target, entry.centroids[clusterIdx]);
                    float minDist = (float) Math.sqrt(centroidDist) - (float) Math.sqrt(entry.clusterRadii[clusterIdx]);
                    if (minDist > 0) {
                        float bestPossibleScore = 1.0f / (1.0f + minDist * minDist);
                        if (bestPossibleScore <= kthBestScore) {
                            break;
                        }
                    }
                }

                clusterData.seek(entry.clusterOffsets[clusterIdx]);
                int missStreak = 0;
                int missThreshold = Math.max(16, clusterSize / 4);

                for (int i = 0; i < clusterSize; i++) {
                    int docId = clusterData.readInt();
                    clusterData.readBytes(storedBytes, 0, entry.dimension);
                    float storedCorrection = Float.intBitsToFloat(clusterData.readInt());

                    if (acceptDocs != null && acceptDocs.get(docId) == false) continue;

                    float approxScore = sqScorer.score(queryBytes, queryCorrection, storedBytes, storedCorrection);
                    knnCollector.incVisitedCount(1);

                    if (topCandidates.size() < candidateCapacity) {
                        topCandidates.add(new int[] { docId, Float.floatToIntBits(approxScore) });
                        if (topCandidates.size() == knnCollector.k()) {
                            kthBestScore = Float.intBitsToFloat(topCandidates.peek()[1]);
                        }
                    } else if (approxScore > kthBestScore) {
                        topCandidates.poll();
                        topCandidates.add(new int[] { docId, Float.floatToIntBits(approxScore) });
                        kthBestScore = Float.intBitsToFloat(topCandidates.peek()[1]);
                        missStreak = 0;
                    } else {
                        missStreak++;
                    }

                    // Intra-cluster early termination: vectors are sorted by centroid distance,
                    // so consecutive misses suggest remaining vectors won't be competitive
                    if (topCandidates.size() >= knnCollector.k() && missStreak >= missThreshold) {
                        break;
                    }
                }
            }
        }

        if (topCandidates.isEmpty()) {
            return;
        }

        // Convert to sorted list (descending by score)
        List<int[]> candidates = new ArrayList<>(topCandidates.size());
        while (topCandidates.isEmpty() == false) {
            candidates.add(topCandidates.poll());
        }
        Collections.reverse(candidates);

        int resultCount = Math.min(candidates.size(), knnCollector.k());

        if (rerank) {
            // Phase 2: rerank top candidates with exact vectors
            int bytesPerEntry = Integer.BYTES + entry.dimension * Float.BYTES;
            try (IndexInput rawData = data.clone()) {
                float[] vector = new float[entry.dimension];
                for (int i = 0; i < resultCount; i++) {
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
                        float approxScore = Float.intBitsToFloat(candidates.get(i)[1]);
                        knnCollector.collect(docId, approxScore);
                    }
                }
            }
        } else {
            for (int i = 0; i < resultCount; i++) {
                int docId = candidates.get(i)[0];
                float approxScore = Float.intBitsToFloat(candidates.get(i)[1]);
                knnCollector.collect(docId, approxScore);
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
            if (entry.clusterOffsets != null) {
                bytes += (long) entry.clusterOffsets.length * Long.BYTES;
            }
            if (entry.clusterSizes != null) {
                bytes += (long) entry.clusterSizes.length * Integer.BYTES;
            }
            if (entry.clusterRadii != null) {
                bytes += (long) entry.clusterRadii.length * Float.BYTES;
            }
            if (entry.centroidGraph != null) {
                bytes += (long) entry.nlist * 16 * 2 * Integer.BYTES;
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
        byte sqBits,
        float[][] centroids,
        ScalarQuantizer scalarQuantizer,
        long[] clusterOffsets,
        int[] clusterSizes,
        float[] clusterRadii,
        long rawVectorDataOffset,
        long rawVectorDataLength,
        long flatDataOffset,
        long flatDataLength,
        OnHeapHnswGraph centroidGraph,
        RandomAccessVectorValues.Floats centroidValues
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
