/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.mapper;

import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
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

public class IvfPqKnnQueryBuilder extends AbstractQueryBuilder<IvfPqKnnQueryBuilder> {

    public static final String NAME = "ivfpq_knn";
    private static final ParseField FIELD_FIELD = new ParseField("field");
    private static final ParseField QUERY_VECTOR_FIELD = new ParseField("query_vector");
    private static final ParseField K_FIELD = new ParseField("k");
    private static final ParseField NUM_CANDIDATES_FIELD = new ParseField("num_candidates");

    private static final ObjectParser<IvfPqKnnQueryBuilder, Void> PARSER = new ObjectParser<>(NAME, IvfPqKnnQueryBuilder::new);

    static {
        PARSER.declareString(IvfPqKnnQueryBuilder::setField, FIELD_FIELD);
        PARSER.declareFloatArray(IvfPqKnnQueryBuilder::setQueryVector, QUERY_VECTOR_FIELD);
        PARSER.declareInt(IvfPqKnnQueryBuilder::setK, K_FIELD);
        PARSER.declareInt(IvfPqKnnQueryBuilder::setNumCandidates, NUM_CANDIDATES_FIELD);
        declareStandardFields(PARSER);
    }

    private String field;
    private float[] queryVector;
    private int k = 10;
    private int numCandidates = 100;

    public IvfPqKnnQueryBuilder() {}

    public IvfPqKnnQueryBuilder(String field, float[] queryVector, int k) {
        this.field = Objects.requireNonNull(field);
        this.queryVector = Objects.requireNonNull(queryVector);
        this.k = k;
        this.numCandidates = Math.max(k, 100);
    }

    public IvfPqKnnQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.field = in.readString();
        this.queryVector = in.readFloatArray();
        this.k = in.readVInt();
        this.numCandidates = in.readVInt();
    }

    public static IvfPqKnnQueryBuilder fromXContent(XContentParser parser) throws IOException {
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
        this.k = k;
    }

    private void setNumCandidates(int numCandidates) {
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
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        builder.array(QUERY_VECTOR_FIELD.getPreferredName(), queryVector);
        builder.field(K_FIELD.getPreferredName(), k);
        builder.field(NUM_CANDIDATES_FIELD.getPreferredName(), numCandidates);
        boostAndQueryNameToXContent(builder);
        builder.endObject();
    }

    @Override
    protected Query doToQuery(SearchExecutionContext context) throws IOException {
        MappedFieldType fieldType = context.getFieldType(field);
        if (fieldType == null) {
            throw new IllegalArgumentException("field [" + field + "] does not exist in the mapping");
        }
        if (fieldType instanceof IvfPqVectorFieldMapper.IvfPqVectorFieldType == false) {
            throw new IllegalArgumentException(
                "[" + NAME + "] query only supports [" + IvfPqVectorFieldMapper.CONTENT_TYPE + "] fields"
            );
        }
        return new KnnFloatVectorQuery(field, queryVector, numCandidates);
    }

    @Override
    protected boolean doEquals(IvfPqKnnQueryBuilder other) {
        return Objects.equals(field, other.field)
            && Arrays.equals(queryVector, other.queryVector)
            && k == other.k
            && numCandidates == other.numCandidates;
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(field, Arrays.hashCode(queryVector), k, numCandidates);
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
