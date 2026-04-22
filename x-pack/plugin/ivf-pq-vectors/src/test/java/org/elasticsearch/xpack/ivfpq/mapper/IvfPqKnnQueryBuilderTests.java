/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq.mapper;

import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.elasticsearch.common.Strings;
import org.elasticsearch.common.compress.CompressedXContent;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.test.AbstractQueryTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;
import org.elasticsearch.xpack.ivfpq.LocalStateIvfPqVectors;

import java.io.IOException;
import java.util.Collection;
import java.util.List;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;

public class IvfPqKnnQueryBuilderTests extends AbstractQueryTestCase<IvfPqKnnQueryBuilder> {

    private static final String VECTOR_FIELD = "ivfpq_field";
    private static final int DIMS = 8;

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return List.of(LocalStateIvfPqVectors.class);
    }

    @Override
    protected void initializeAdditionalMappings(MapperService mapperService) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "ivfpq_vector")
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
    protected IvfPqKnnQueryBuilder doCreateTestQueryBuilder() {
        float[] vector = new float[DIMS];
        for (int i = 0; i < DIMS; i++) {
            vector[i] = randomFloat();
        }
        int k = randomIntBetween(1, 50);
        return new IvfPqKnnQueryBuilder(VECTOR_FIELD, vector, k);
    }

    @Override
    protected void doAssertLuceneQuery(IvfPqKnnQueryBuilder queryBuilder, Query query, SearchExecutionContext context) {
        assertThat(query, instanceOf(KnnFloatVectorQuery.class));
    }

    public void testNonexistentField() {
        float[] vector = new float[DIMS];
        IvfPqKnnQueryBuilder qb = new IvfPqKnnQueryBuilder("nonexistent", vector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("does not exist in the mapping"));
    }

    public void testWrongFieldType() throws IOException {
        float[] vector = new float[DIMS];
        IvfPqKnnQueryBuilder qb = new IvfPqKnnQueryBuilder("mapped_string", vector, 10);
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> qb.toQuery(createSearchExecutionContext()));
        assertThat(e.getMessage(), containsString("only supports [ivfpq_vector] fields"));
    }
}
