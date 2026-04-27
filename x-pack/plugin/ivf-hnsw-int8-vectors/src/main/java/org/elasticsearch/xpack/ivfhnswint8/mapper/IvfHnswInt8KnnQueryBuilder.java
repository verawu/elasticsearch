/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8.mapper;

import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.elasticsearch.xpack.ivfhnswint8.codec.IvfHnswInt8VectorsReader;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.query.AbstractQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.xcontent.ObjectParser;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class IvfHnswInt8KnnQueryBuilder extends AbstractQueryBuilder<IvfHnswInt8KnnQueryBuilder> {

    public static final String NAME = "ivf_hnsw_int8_knn";
    private static final ParseField FIELD_FIELD = new ParseField("field");
    private static final ParseField QUERY_VECTOR_FIELD = new ParseField("query_vector");
    private static final ParseField K_FIELD = new ParseField("k");
    private static final ParseField NUM_CANDIDATES_FIELD = new ParseField("num_candidates");
    private static final ParseField NPROBE_FIELD = new ParseField("nprobe");

    private static final ObjectParser<IvfHnswInt8KnnQueryBuilder, Void> PARSER = new ObjectParser<>(NAME, IvfHnswInt8KnnQueryBuilder::new);

    static {
        PARSER.declareString(IvfHnswInt8KnnQueryBuilder::setField, FIELD_FIELD);
        PARSER.declareFloatArray(IvfHnswInt8KnnQueryBuilder::setQueryVector, QUERY_VECTOR_FIELD);
        PARSER.declareInt(IvfHnswInt8KnnQueryBuilder::setK, K_FIELD);
        PARSER.declareInt(IvfHnswInt8KnnQueryBuilder::setNumCandidates, NUM_CANDIDATES_FIELD);
        PARSER.declareInt(IvfHnswInt8KnnQueryBuilder::setNprobe, NPROBE_FIELD);
        declareStandardFields(PARSER);
    }

    private String field;
    private float[] queryVector;
    private int k = 10;
    private int numCandidates = 100;
    private Integer nprobeOverride;

    public IvfHnswInt8KnnQueryBuilder() {}

    public IvfHnswInt8KnnQueryBuilder(String field, float[] queryVector, int k) {
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        if (k < 1) {
            throw new IllegalArgumentException("[k] must be greater than 0, got " + k);
        }
        this.k = k;
        this.numCandidates = Math.max(k, 100);
    }

    public IvfHnswInt8KnnQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.field = in.readString();
        this.queryVector = in.readFloatArray();
        this.k = in.readVInt();
        this.numCandidates = in.readVInt();
        this.nprobeOverride = in.readOptionalVInt();
    }

    public static IvfHnswInt8KnnQueryBuilder fromXContent(XContentParser parser) throws IOException {
        return PARSER.parse(parser, null);
    }

    public String getField() {
        return field;
    }

    public float[] getQueryVector() {
        return queryVector;
    }

    public int getK() {
        return k;
    }

    public int getNumCandidates() {
        return numCandidates;
    }

    public Integer getNprobeOverride() {
        return nprobeOverride;
    }

    private void setNprobe(int nprobe) {
        if (nprobe < 1) {
            throw new IllegalArgumentException("[nprobe] must be greater than 0, got " + nprobe);
        }
        this.nprobeOverride = nprobe;
    }

    private void setField(String field) {
        this.field = field;
    }

    private void setQueryVector(List<Float> queryVector) {
        this.queryVector = new float[queryVector.size()];
        for (int i = 0; i < queryVector.size(); i++) {
            this.queryVector[i] = queryVector.get(i);
        }
    }

    private void setK(int k) {
        if (k < 1) {
            throw new IllegalArgumentException("[k] must be greater than 0, got " + k);
        }
        this.k = k;
    }

    private void setNumCandidates(int numCandidates) {
        if (numCandidates < 1) {
            throw new IllegalArgumentException("[num_candidates] must be greater than 0, got " + numCandidates);
        }
        if (numCandidates > 10_000) {
            throw new IllegalArgumentException("[num_candidates] cannot exceed 10000, got " + numCandidates);
        }
        this.numCandidates = numCandidates;
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(field);
        out.writeFloatArray(queryVector);
        out.writeVInt(k);
        out.writeVInt(numCandidates);
        out.writeOptionalVInt(nprobeOverride);
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        builder.array(QUERY_VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(K_FIELD.getPreferredName(), k);
        builder.field(NUM_CANDIDATES_FIELD.getPreferredName(), numCandidates);
        if (nprobeOverride != null) {
            builder.field(NPROBE_FIELD.getPreferredName(), nprobeOverride);
        }
        boostAndQueryNameToXContent(builder);
        builder.endObject();
    }

    @Override
    protected Query doToQuery(SearchExecutionContext context) throws IOException {
        MappedFieldType fieldType = context.getFieldType(field);
        if (fieldType == null) {
            return new MatchNoDocsQuery("field [" + field + "] does not exist in the mapping");
        }
        if (fieldType instanceof IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType == false) {
            throw new IllegalArgumentException(
                "[" + NAME + "] query only supports [" + IvfHnswInt8VectorFieldMapper.CONTENT_TYPE + "] fields"
            );
        }
        IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType ivfFieldType =
            (IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType) fieldType;
        if (queryVector.length != ivfFieldType.getDims()) {
            throw new IllegalArgumentException(
                "the query vector has a different dimension [" + queryVector.length
                    + "] than the index vectors [" + ivfFieldType.getDims() + "]"
            );
        }
        for (int i = 0; i < queryVector.length; i++) {
            if (Float.isNaN(queryVector[i])) {
                throw new IllegalArgumentException("query vector contains NaN at dimension [" + i + "]");
            }
            if (Float.isInfinite(queryVector[i])) {
                throw new IllegalArgumentException("query vector contains infinite value at dimension [" + i + "]");
            }
        }
        if (numCandidates < k) {
            throw new IllegalArgumentException("[num_candidates] cannot be less than [k]");
        }
        if (nprobeOverride != null) {
            IvfHnswInt8VectorsReader.setNprobeOverride(nprobeOverride);
        }
        try {
            return new KnnFloatVectorQuery(field, queryVector, numCandidates);
        } finally {
            IvfHnswInt8VectorsReader.clearNprobeOverride();
        }
    }

    @Override
    protected boolean doEquals(IvfHnswInt8KnnQueryBuilder other) {
        return Objects.equals(field, other.field)
            && Arrays.equals(queryVector, other.queryVector)
            && k == other.k
            && numCandidates == other.numCandidates
            && Objects.equals(nprobeOverride, other.nprobeOverride);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(field, Arrays.hashCode(queryVector), k, numCandidates, nprobeOverride);
    }

    @Override
    protected QueryBuilder doRewrite(QueryRewriteContext queryRewriteContext) throws IOException {
        return this;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersions.V_8_18_0;
    }
}
