/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.mapper;

import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.elasticsearch.index.codec.KnnVectorsFormatProvider;
import org.elasticsearch.index.mapper.DocumentParserContext;
import org.elasticsearch.index.mapper.FieldMapper;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.index.mapper.MapperBuilderContext;
import org.elasticsearch.index.mapper.SimpleMappedFieldType;
import org.elasticsearch.index.mapper.SourceValueFetcher;
import org.elasticsearch.index.mapper.TextSearchInfo;
import org.elasticsearch.index.mapper.ValueFetcher;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xpack.ivfpq.codec.IvfPqVectorsFormat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class IvfPqVectorFieldMapper extends FieldMapper implements KnnVectorsFormatProvider {

    public static final String CONTENT_TYPE = "ivfpq_vector";
    private static final int MAX_DIMS = 4096;

    private final int dims;
    private final VectorSimilarityFunction similarity;
    private final int nlist;
    private final int nprobe;
    private final int sqBits;
    private final int trainingThreshold;
    private final boolean rerank;

    private IvfPqVectorFieldMapper(
        String simpleName,
        MappedFieldType mappedFieldType,
        BuilderParams builderParams,
        int dims,
        VectorSimilarityFunction similarity,
        int nlist,
        int nprobe,
        int sqBits,
        int trainingThreshold,
        boolean rerank
    ) {
        super(simpleName, mappedFieldType, builderParams);
        this.dims = dims;
        this.similarity = similarity;
        this.nlist = nlist;
        this.nprobe = nprobe;
        this.sqBits = sqBits;
        this.trainingThreshold = trainingThreshold;
        this.rerank = rerank;
    }

    @Override
    public KnnVectorsFormat getKnnVectorsFormatForField(KnnVectorsFormat defaultFormat) {
        IvfPqVectorsFormat format = new IvfPqVectorsFormat(nlist, nprobe, sqBits, trainingThreshold, 20, rerank);
        return new KnnVectorsFormat(format.getName()) {
            @Override
            public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
                return format.fieldsWriter(state);
            }

            @Override
            public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
                return format.fieldsReader(state);
            }

            @Override
            public int getMaxDimensions(String fieldName) {
                return MAX_DIMS;
            }
        };
    }

    @Override
    public boolean parsesArrayValue() {
        return true;
    }

    @Override
    protected void parseCreateField(DocumentParserContext context) throws IOException {
        XContentParser parser = context.parser();
        if (parser.currentToken() == XContentParser.Token.VALUE_NULL) {
            return;
        }
        float[] vector = new float[dims];
        int i = 0;
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            if (i >= dims) {
                throw new IllegalArgumentException("Vector has more than [" + dims + "] dimensions");
            }
            vector[i++] = parser.floatValue();
        }
        if (i != dims) {
            throw new IllegalArgumentException("Vector has [" + i + "] dimensions, expected [" + dims + "]");
        }

        for (int j = 0; j < dims; j++) {
            if (Float.isNaN(vector[j])) {
                throw new IllegalArgumentException(
                    "vector value at dimension [" + j + "] is NaN, preview of invalid vector: " + vectorPreview(vector)
                );
            }
            if (Float.isInfinite(vector[j])) {
                throw new IllegalArgumentException(
                    "vector value at dimension [" + j + "] is infinite, preview of invalid vector: " + vectorPreview(vector)
                );
            }
        }

        if (similarity == VectorSimilarityFunction.COSINE) {
            float magnitude = 0;
            for (float v : vector) {
                magnitude += v * v;
            }
            magnitude = (float) Math.sqrt(magnitude);
            if (magnitude == 0.0f) {
                throw new IllegalArgumentException(
                    "The [cosine] similarity does not support vectors with zero magnitude"
                );
            }
            for (int d = 0; d < dims; d++) {
                vector[d] /= magnitude;
            }
        }

        IndexableField field = new KnnFloatVectorField(fullPath(), vector, similarity);
        context.doc().addWithKey(fullPath(), field);
    }

    private static String vectorPreview(float[] vector) {
        StringBuilder sb = new StringBuilder("[");
        int previewLen = Math.min(5, vector.length);
        for (int i = 0; i < previewLen; i++) {
            if (i > 0) sb.append(", ");
            sb.append(vector[i]);
        }
        if (vector.length > 5) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }

    @Override
    protected String contentType() {
        return CONTENT_TYPE;
    }

    @Override
    public FieldMapper.Builder getMergeBuilder() {
        return new Builder(leafName()).init(this);
    }

    public static class IvfPqVectorFieldType extends SimpleMappedFieldType {

        private final int dims;
        private final VectorSimilarityFunction similarity;

        public IvfPqVectorFieldType(String name, int dims, VectorSimilarityFunction similarity, Map<String, String> meta) {
            super(name, true, false, false, TextSearchInfo.NONE, meta);
            this.dims = dims;
            this.similarity = similarity;
        }

        @Override
        public String typeName() {
            return CONTENT_TYPE;
        }

        @Override
        public ValueFetcher valueFetcher(SearchExecutionContext context, String format) {
            return SourceValueFetcher.identity(name(), context, format);
        }

        @Override
        public Query existsQuery(SearchExecutionContext context) {
            return new FieldExistsQuery(name());
        }

        @Override
        public Query termQuery(Object value, SearchExecutionContext context) {
            throw new IllegalArgumentException("Field [" + name() + "] of type [" + typeName() + "] does not support term queries");
        }

        public int getDims() {
            return dims;
        }

        public VectorSimilarityFunction getSimilarity() {
            return similarity;
        }
    }

    public static class Builder extends FieldMapper.Builder {

        private final Parameter<Integer> dims = Parameter.intParam("dims", false, b -> ((IvfPqVectorFieldMapper) b).dims, 0)
            .addValidator(v -> {
                if (v <= 0 || v > MAX_DIMS) {
                    throw new IllegalArgumentException("dims must be between 1 and " + MAX_DIMS + ", got " + v);
                }
            });

        private final Parameter<String> similarityParam = Parameter.stringParam("similarity", false, b -> {
            VectorSimilarityFunction sim = ((IvfPqVectorFieldMapper) b).similarity;
            return switch (sim) {
                case EUCLIDEAN -> "l2_norm";
                case DOT_PRODUCT -> "dot_product";
                case COSINE -> "cosine";
                case MAXIMUM_INNER_PRODUCT -> "max_inner_product";
            };
        }, "l2_norm");

        private final Parameter<Integer> nlistParam = Parameter.intParam("nlist", true, b -> ((IvfPqVectorFieldMapper) b).nlist, 256);

        private final Parameter<Integer> nprobeParam = Parameter.intParam("nprobe", true, b -> ((IvfPqVectorFieldMapper) b).nprobe, 8);

        private final Parameter<Integer> sqBitsParam = Parameter.intParam("sq_bits", false, b -> ((IvfPqVectorFieldMapper) b).sqBits, 7);

        private final Parameter<Integer> trainingThresholdParam = Parameter.intParam(
            "training_threshold",
            true,
            b -> ((IvfPqVectorFieldMapper) b).trainingThreshold,
            1000
        );

        private final Parameter<Boolean> rerankParam = Parameter.boolParam(
            "rerank",
            true,
            b -> ((IvfPqVectorFieldMapper) b).rerank,
            false
        );

        private final Parameter<Map<String, String>> meta = Parameter.metaParam();

        public Builder(String name) {
            super(name);
        }

        @Override
        protected Parameter<?>[] getParameters() {
            return new Parameter<?>[] {
                dims, similarityParam, nlistParam, nprobeParam, sqBitsParam, trainingThresholdParam, rerankParam, meta
            };
        }

        private VectorSimilarityFunction parseSimilarity(String value) {
            return switch (value) {
                case "l2_norm" -> VectorSimilarityFunction.EUCLIDEAN;
                case "dot_product" -> VectorSimilarityFunction.DOT_PRODUCT;
                case "cosine" -> VectorSimilarityFunction.COSINE;
                case "max_inner_product" -> VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
                default -> throw new IllegalArgumentException("Unknown similarity [" + value + "]");
            };
        }

        @Override
        public IvfPqVectorFieldMapper build(MapperBuilderContext context) {
            int dimsVal = dims.getValue();
            int nlistVal = nlistParam.getValue();
            int nprobeVal = nprobeParam.getValue();
            int sqBitsVal = sqBitsParam.getValue();
            int trainingThresholdVal = trainingThresholdParam.getValue();
            VectorSimilarityFunction sim = parseSimilarity(similarityParam.getValue());

            if (nlistVal <= 0) {
                throw new IllegalArgumentException("nlist must be > 0, got " + nlistVal);
            }
            if (nprobeVal <= 0) {
                throw new IllegalArgumentException("nprobe must be > 0, got " + nprobeVal);
            }
            if (trainingThresholdVal <= 0) {
                throw new IllegalArgumentException("training_threshold must be > 0, got " + trainingThresholdVal);
            }
            if (nprobeVal > nlistVal) {
                throw new IllegalArgumentException("nprobe [" + nprobeVal + "] must be <= nlist [" + nlistVal + "]");
            }
            if (sqBitsVal != 4 && sqBitsVal != 7 && sqBitsVal != 8) {
                throw new IllegalArgumentException("sq_bits must be 4, 7, or 8, got " + sqBitsVal);
            }

            IvfPqVectorFieldType fieldType = new IvfPqVectorFieldType(
                context.buildFullName(leafName()),
                dimsVal,
                sim,
                meta.getValue()
            );

            return new IvfPqVectorFieldMapper(
                leafName(),
                fieldType,
                builderParams(this, context),
                dimsVal,
                sim,
                nlistVal,
                nprobeVal,
                sqBitsVal,
                trainingThresholdVal,
                rerankParam.getValue()
            );
        }
    }
}
