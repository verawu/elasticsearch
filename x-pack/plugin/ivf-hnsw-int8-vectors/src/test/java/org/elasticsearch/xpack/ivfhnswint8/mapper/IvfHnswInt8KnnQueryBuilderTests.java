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
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.compress.CompressedXContent;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.test.AbstractQueryTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xpack.ivfhnswint8.LocalStateIvfHnswInt8Vectors;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;

public class IvfHnswInt8KnnQueryBuilderTests extends AbstractQueryTestCase<IvfHnswInt8KnnQueryBuilder> {

    private static final String VECTOR_FIELD = "ivf_hnsw_int8_field";
    private static final int DIMS = 8;

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return List.of(LocalStateIvfHnswInt8Vectors.class);
    }

    @Override
    protected void initializeAdditionalMappings(MapperService mapperService) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "ivf_hnsw_int8_vector")
            .field("dims", DIMS)
            .endObject()
            .endObject()
            .endObject();
        mapperService.merge(
            MapperService.SINGLE_MAPPING_NAME,
            new CompressedXContent(Strings.toString(builder)),
            MapperService.MergeReason.MAPPING_UPDATE
        );
    }

    @Override
    protected IvfHnswInt8KnnQueryBuilder doCreateTestQueryBuilder() {
        float[] vector = new float[DIMS];
        for (int i = 0; i < DIMS; i++) {
            vector[i] = randomFloat();
        }
        int k = randomIntBetween(1, 50);
        return new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, vector, k);
    }

    @Override
    protected void doAssertLuceneQuery(IvfHnswInt8KnnQueryBuilder queryBuilder, Query query, SearchExecutionContext context) {
        assertThat(query, instanceOf(KnnFloatVectorQuery.class));
    }

    public void testNonexistentField() throws IOException {
        float[] vector = new float[DIMS];
        IvfHnswInt8KnnQueryBuilder qb = new IvfHnswInt8KnnQueryBuilder("nonexistent", vector, 10);
        Query query = qb.toQuery(createSearchExecutionContext());
        assertThat(query, instanceOf(MatchNoDocsQuery.class));
    }

    public void testWrongFieldType() throws IOException {
        float[] vector = new float[DIMS];
        IvfHnswInt8KnnQueryBuilder qb = new IvfHnswInt8KnnQueryBuilder("mapped_string", vector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("only supports [ivf_hnsw_int8_vector] fields"));
    }

    public void testDimensionMismatch() {
        float[] wrongDimsVector = new float[DIMS + 2];
        IvfHnswInt8KnnQueryBuilder qb = new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, wrongDimsVector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("different dimension"));
    }

    public void testNaNQueryVector() {
        float[] vector = new float[DIMS];
        vector[2] = Float.NaN;
        IvfHnswInt8KnnQueryBuilder qb = new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, vector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("NaN"));
    }

    public void testInfiniteQueryVector() {
        float[] vector = new float[DIMS];
        vector[0] = Float.POSITIVE_INFINITY;
        IvfHnswInt8KnnQueryBuilder qb = new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, vector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("infinite"));
    }

    public void testKZero() {
        float[] vector = new float[DIMS];
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, vector, 0));
    }

    public void testKNegative() {
        float[] vector = new float[DIMS];
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8KnnQueryBuilder(VECTOR_FIELD, vector, -1));
    }
}
