/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.engine;

import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.search.vectors.KnnVectorQueryBuilder;
import org.elasticsearch.test.ESIntegTestCase;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentFactory;

import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.greaterThan;

public class DeferredVectorBuildIT extends ESIntegTestCase {

    private static final String INDEX_NAME = "deferred-vector-test";
    private static final String VECTOR_FIELD = "vector";
    private static final int DIMS = 4;

    private Settings indexSettings(boolean deferred) {
        return Settings.builder()
            .put("index.number_of_shards", 1)
            .put("index.number_of_replicas", 0)
            .put("index.vector_build.deferred", deferred)
            .build();
    }

    private XContentBuilder hnswMapping(String type) throws Exception {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(VECTOR_FIELD)
            .field("type", "dense_vector")
            .field("dims", DIMS)
            .field("index", true)
            .field("similarity", "l2_norm")
            .startObject("index_options")
            .field("type", type)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }

    public void testEndToEndIndexAndSearch() throws Exception {
        indicesAdmin().prepareCreate(INDEX_NAME).setSettings(indexSettings(true)).setMapping(hnswMapping("hnsw")).get();
        int docCount = 100;
        for (int i = 0; i < docCount; i++) {
            float val = i * 0.01f;
            prepareIndex(INDEX_NAME).setSource(
                XContentFactory.jsonBuilder()
                    .startObject()
                    .field(VECTOR_FIELD, new float[] { val, val, val, val })
                    .endObject()
            ).get();
        }
        refresh(INDEX_NAME);

        float[] query = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
        SearchResponse resp = client().prepareSearch(INDEX_NAME)
            .setQuery(new KnnVectorQueryBuilder(VECTOR_FIELD, query, 10, 100, null, null))
            .setSize(10)
            .get();

        assertThat(resp.getHits().getHits().length, greaterThan(0));
        assertThat(resp.getHits().getHits().length, equalTo(10));
    }

    public void testDeferredMatchesSync() throws Exception {
        String deferredIdx = "deferred-idx";
        String syncIdx = "sync-idx";

        indicesAdmin().prepareCreate(deferredIdx).setSettings(indexSettings(true)).setMapping(hnswMapping("hnsw")).get();
        indicesAdmin().prepareCreate(syncIdx).setSettings(indexSettings(false)).setMapping(hnswMapping("hnsw")).get();

        int docCount = 50;
        for (int i = 0; i < docCount; i++) {
            float val = i * 0.02f;
            XContentBuilder source = XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, new float[] { val, val, val, val })
                .endObject();
            prepareIndex(deferredIdx).setSource(source).get();
            source = XContentFactory.jsonBuilder()
                .startObject()
                .field(VECTOR_FIELD, new float[] { val, val, val, val })
                .endObject();
            prepareIndex(syncIdx).setSource(source).get();
        }
        refresh(deferredIdx, syncIdx);

        float[] query = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
        int k = 10;

        SearchResponse deferredResp = client().prepareSearch(deferredIdx)
            .setQuery(new KnnVectorQueryBuilder(VECTOR_FIELD, query, k, 100, null, null))
            .setSize(k)
            .get();

        SearchResponse syncResp = client().prepareSearch(syncIdx)
            .setQuery(new KnnVectorQueryBuilder(VECTOR_FIELD, query, k, 100, null, null))
            .setSize(k)
            .get();

        assertThat(deferredResp.getHits().getHits().length, equalTo(k));
        assertThat(syncResp.getHits().getHits().length, equalTo(k));

        for (int i = 0; i < k; i++) {
            assertEquals(
                "rank " + i + " scores should match",
                syncResp.getHits().getHits()[i].getScore(),
                deferredResp.getHits().getHits()[i].getScore(),
                0.01f
            );
        }
    }

    public void testAllHnswTypes() throws Exception {
        String[] types = { "hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw" };
        for (String type : types) {
            String idxName = "test-" + type.replace("_", "-");
            indicesAdmin().prepareCreate(idxName).setSettings(indexSettings(true)).setMapping(hnswMapping(type)).get();

            int docCount = 30;
            for (int i = 0; i < docCount; i++) {
                float val = i * 0.03f;
                prepareIndex(idxName).setSource(
                    XContentFactory.jsonBuilder()
                        .startObject()
                        .field(VECTOR_FIELD, new float[] { val, val, val, val })
                        .endObject()
                ).get();
            }
            refresh(idxName);

            float[] query = new float[] { 0.5f, 0.5f, 0.5f, 0.5f };
            SearchResponse resp = client().prepareSearch(idxName)
                .setQuery(new KnnVectorQueryBuilder(VECTOR_FIELD, query, 5, 50, null, null))
                .setSize(5)
                .get();

            assertThat("type=" + type + " should return results", resp.getHits().getHits().length, greaterThan(0));
        }
    }
}
