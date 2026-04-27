/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8.mapper;

import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.Query;
import org.elasticsearch.index.mapper.LuceneDocument;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.MapperParsingException;
import org.elasticsearch.index.mapper.MapperService;
import org.elasticsearch.index.mapper.MapperTestCase;
import org.elasticsearch.index.mapper.ParsedDocument;
import org.elasticsearch.index.mapper.SourceToParse;
import org.elasticsearch.index.mapper.ValueFetcher;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.search.lookup.Source;
import org.elasticsearch.search.lookup.SourceProvider;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xpack.ivfhnswint8.LocalStateIvfHnswInt8Vectors;
import org.junit.AssumptionViolatedException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.instanceOf;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class IvfHnswInt8VectorFieldMapperTests extends MapperTestCase {

    @Override
    protected Collection<? extends Plugin> getPlugins() {
        return Collections.singletonList(new LocalStateIvfHnswInt8Vectors(SETTINGS));
    }

    @Override
    protected void minimalMapping(XContentBuilder b) throws IOException {
        b.field("type", "ivf_hnsw_int8_vector").field("dims", 8);
    }

    @Override
    protected Object getSampleValueForDocument() {
        return List.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5);
    }

    @Override
    protected void registerParameters(ParameterChecker checker) throws IOException {
        checker.registerConflictCheck(
            "dims",
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8)),
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 16))
        );
        checker.registerConflictCheck(
            "similarity",
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("similarity", "l2_norm")),
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("similarity", "dot_product"))
        );
        checker.registerConflictCheck(
            "sq_bits",
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("sq_bits", 7)),
            fieldMapping(b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("sq_bits", 8))
        );
        checker.registerUpdateCheck(
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nlist", 256),
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nlist", 128),
            m -> {}
        );
        checker.registerUpdateCheck(
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nprobe", 16),
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nprobe", 8),
            m -> {}
        );
        checker.registerUpdateCheck(
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("training_threshold", 1000),
            b -> b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("training_threshold", 500),
            m -> {}
        );
    }

    @Override
    protected boolean supportsStoredFields() {
        return false;
    }

    @Override
    protected boolean supportsIgnoreMalformed() {
        return false;
    }

    @Override
    protected boolean supportsEmptyInputArray() {
        return false;
    }

    @Override
    protected void assertSearchable(MappedFieldType fieldType) {
        assertThat(fieldType, instanceOf(IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType.class));
        assertTrue(fieldType.isIndexed());
        assertTrue(fieldType.isSearchable());
    }

    @Override
    protected void assertExistsQuery(MappedFieldType fieldType, Query query, LuceneDocument fields) {
        assertThat(query, instanceOf(FieldExistsQuery.class));
        FieldExistsQuery existsQuery = (FieldExistsQuery) query;
        assertEquals("field", existsQuery.getField());
    }

    @Override
    public void testAggregatableConsistency() {}

    @Override
    protected IngestScriptSupport ingestScriptSupport() {
        throw new AssumptionViolatedException("not supported");
    }

    @Override
    protected SyntheticSourceSupport syntheticSourceSupport(boolean ignoreMalformed) {
        throw new AssumptionViolatedException("ivf_hnsw_int8_vector does not support synthetic source");
    }

    @Override
    protected void assertFetchMany(MapperService mapperService, String field, Object value, String format, int count) throws IOException {
        assumeFalse("Dense vectors currently don't support multiple values in the same field", false);
    }

    @Override
    protected void assertFetch(MapperService mapperService, String field, Object value, String format) throws IOException {
        MappedFieldType ft = mapperService.fieldType(field);
        MappedFieldType.FielddataOperation fdt = MappedFieldType.FielddataOperation.SEARCH;
        SourceToParse source = source(b -> b.field(ft.name(), value));
        SearchExecutionContext searchExecutionContext = mock(SearchExecutionContext.class);
        when(searchExecutionContext.isSourceEnabled()).thenReturn(true);
        when(searchExecutionContext.sourcePath(field)).thenReturn(Set.of(field));
        when(searchExecutionContext.getForField(ft, fdt)).thenAnswer(inv -> fieldDataLookup(mapperService).apply(ft, () -> {
            throw new UnsupportedOperationException();
        }, fdt));
        ValueFetcher nativeFetcher = ft.valueFetcher(searchExecutionContext, format);
        ParsedDocument doc = mapperService.documentMapper().parse(source);
        withLuceneIndex(mapperService, iw -> iw.addDocuments(doc.docs()), ir -> {
            Source s = SourceProvider.fromLookup(mapperService.mappingLookup(), null, mapperService.getMapperMetrics().sourceFieldMetrics())
                .getSource(ir.leaves().get(0), 0);
            nativeFetcher.setNextReader(ir.leaves().get(0));
            List<Object> fromNative = nativeFetcher.fetchValues(s, 0, new ArrayList<>());
            assertFalse("Should fetch at least one value", fromNative.isEmpty());
        });
    }

    @Override
    protected void randomFetchTestFieldConfig(XContentBuilder b) throws IOException {
        b.field("type", "ivf_hnsw_int8_vector").field("dims", 8);
    }

    @Override
    protected Object generateRandomInputValue(MappedFieldType ft) {
        IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType vectorFieldType = (IvfHnswInt8VectorFieldMapper.IvfHnswInt8VectorFieldType) ft;
        int dims = vectorFieldType.getDims();
        List<Double> vector = new ArrayList<>(dims);
        for (int i = 0; i < dims; i++) {
            vector.add((double) randomFloat());
        }
        return vector;
    }

    // --- Validation error tests ---

    public void testDimsZero() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 0);
        })));
        assertThat(e.getMessage(), containsString("dims must be between 1 and 4096"));
    }

    public void testDimsExceedsMax() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 5000);
        })));
        assertThat(e.getMessage(), containsString("dims must be between 1 and 4096"));
    }

    public void testNprobeExceedsNlist() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nlist", 4).field("nprobe", 8);
        })));
        assertThat(e.getMessage(), containsString("nprobe [8] must be <= nlist [4]"));
    }

    public void testInvalidSqBits() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("sq_bits", 3);
        })));
        assertThat(e.getMessage(), containsString("sq_bits must be 4, 7, or 8"));
    }

    public void testCannotBeUsedInMultifields() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "keyword");
            b.startObject("fields");
            b.startObject("vectors");
            minimalMapping(b);
            b.endObject();
            b.endObject();
        })));
        assertThat(e.getMessage(), containsString("can't be used in multifields"));
    }

    public void testNaNVector() throws Exception {
        MapperService mapperService = createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 4);
        }));
        Exception e = expectThrows(org.elasticsearch.index.mapper.DocumentParsingException.class, () -> {
            mapperService.documentMapper().parse(source(b -> b.field("field", List.of(1.0, Float.NaN, 0.0, 0.0))));
        });
        assertThat(e.getCause().getMessage(), containsString("NaN"));
    }

    public void testInfinityVector() throws Exception {
        MapperService mapperService = createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 4);
        }));
        Exception e = expectThrows(org.elasticsearch.index.mapper.DocumentParsingException.class, () -> {
            mapperService.documentMapper()
                .parse(source(b -> b.field("field", List.of(1.0, Double.POSITIVE_INFINITY, 0.0, 0.0))));
        });
        assertThat(e.getCause().getMessage(), containsString("infinite"));
    }

    public void testZeroMagnitudeCosineVector() throws Exception {
        MapperService mapperService = createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 4).field("similarity", "cosine");
        }));
        Exception e = expectThrows(org.elasticsearch.index.mapper.DocumentParsingException.class, () -> {
            mapperService.documentMapper().parse(source(b -> b.field("field", List.of(0.0, 0.0, 0.0, 0.0))));
        });
        assertThat(e.getCause().getMessage(), containsString("zero magnitude"));
    }

    public void testNlistZero() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nlist", 0);
        })));
        assertThat(e.getMessage(), containsString("nlist must be > 0"));
    }

    public void testNprobeZero() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("nprobe", 0);
        })));
        assertThat(e.getMessage(), containsString("nprobe must be > 0"));
    }

    public void testTrainingThresholdZero() {
        Exception e = expectThrows(MapperParsingException.class, () -> createMapperService(fieldMapping(b -> {
            b.field("type", "ivf_hnsw_int8_vector").field("dims", 8).field("training_threshold", 0);
        })));
        assertThat(e.getMessage(), containsString("training_threshold must be > 0"));
    }

    @Override
    public void testSyntheticSourceKeepArrays() {}

}
