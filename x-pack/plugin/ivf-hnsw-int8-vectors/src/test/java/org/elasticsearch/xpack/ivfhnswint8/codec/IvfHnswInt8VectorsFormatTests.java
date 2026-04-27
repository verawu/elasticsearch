/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8.codec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.lucene912.Lucene912Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.util.Random;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class IvfHnswInt8VectorsFormatTests extends ESTestCase {

    static {
        LogConfigurator.loadLog4jPlugins();
        LogConfigurator.configureESLogging();
    }

    private Codec getCodec(int nlist, int nprobe, int sqBits, int trainingThreshold) {
        return new Lucene912Codec() {
            @Override
            public KnnVectorsFormat getKnnVectorsFormatForField(String field) {
                return new IvfHnswInt8VectorsFormat(nlist, nprobe, sqBits, trainingThreshold, 20);
            }
        };
    }

    public void testWriteAndReadFlatFallback() throws IOException {
        int dims = 8;
        int nVectors = 10;
        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(16, 4, 7, 100));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                Random random = new Random(42);
                for (int i = 0; i < nVectors; i++) {
                    Document doc = new Document();
                    float[] vector = randomVector(dims, random);
                    doc.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReader leafReader = reader.leaves().get(0).reader();
                FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vec");
                assertNotNull(vectorValues);
                assertEquals(dims, vectorValues.dimension());
                assertEquals(nVectors, vectorValues.size());

                int count = 0;
                while (vectorValues.nextDoc() != NO_MORE_DOCS) {
                    float[] value = vectorValues.vectorValue();
                    assertNotNull(value);
                    assertEquals(dims, value.length);
                    count++;
                }
                assertEquals(nVectors, count);
            }
        }
    }

    public void testWriteAndReadIvfSq() throws IOException {
        int dims = 16;
        int nVectors = 200;
        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 4, 7, 50));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                Random random = new Random(42);
                for (int i = 0; i < nVectors; i++) {
                    Document doc = new Document();
                    float[] vector = randomVector(dims, random);
                    doc.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                LeafReader leafReader = reader.leaves().get(0).reader();
                FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vec");
                assertNotNull(vectorValues);
                assertEquals(dims, vectorValues.dimension());
                assertEquals(nVectors, vectorValues.size());
            }
        }
    }

    public void testSearchFlat() throws IOException {
        int dims = 8;
        int nVectors = 20;
        float[][] vectors = new float[nVectors][dims];
        Random random = new Random(42);
        for (int i = 0; i < nVectors; i++) {
            vectors[i] = randomVector(dims, random);
        }

        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 4, 7, 100));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                float[] query = vectors[0];
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery("vec", query, 5), 5);
                assertTrue("Should return results", topDocs.scoreDocs.length > 0);
                assertEquals(0, topDocs.scoreDocs[0].doc);
            }
        }
    }

    public void testSearchIvfSq() throws IOException {
        int dims = 16;
        int nVectors = 200;
        float[][] vectors = new float[nVectors][dims];
        Random random = new Random(42);
        for (int i = 0; i < nVectors; i++) {
            vectors[i] = randomVector(dims, random);
        }

        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 8, 7, 50));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (float[] vector : vectors) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                float[] query = vectors[0];
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery("vec", query, 10), 10);
                assertTrue("Should return results", topDocs.scoreDocs.length > 0);
                boolean foundExact = false;
                for (var scoreDoc : topDocs.scoreDocs) {
                    if (scoreDoc.doc == 0) {
                        foundExact = true;
                        break;
                    }
                }
                assertTrue("Exact match should be found with full nprobe", foundExact);
            }
        }
    }

    public void testMerge() throws IOException {
        int dims = 16;
        int nVectorsPerSegment = 100;
        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 4, 7, 50));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                Random random = new Random(42);
                for (int seg = 0; seg < 2; seg++) {
                    for (int i = 0; i < nVectorsPerSegment; i++) {
                        Document doc = new Document();
                        doc.add(new KnnFloatVectorField("vec", randomVector(dims, random), VectorSimilarityFunction.EUCLIDEAN));
                        writer.addDocument(doc);
                    }
                    writer.commit();
                }
                writer.forceMerge(1);
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(1, reader.leaves().size());
                LeafReader leafReader = reader.leaves().get(0).reader();
                FloatVectorValues vectorValues = leafReader.getFloatVectorValues("vec");
                assertNotNull(vectorValues);
                assertEquals(nVectorsPerSegment * 2, vectorValues.size());
            }
        }
    }

    public void testDeletedDocsNotInSearchResults() throws IOException {
        int dims = 16;
        int nVectors = 200;
        float[][] vectors = new float[nVectors][dims];
        Random random = new Random(42);
        for (int i = 0; i < nVectors; i++) {
            vectors[i] = randomVector(dims, random);
        }

        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 8, 7, 50));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (int i = 0; i < nVectors; i++) {
                    Document doc = new Document();
                    doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
                    doc.add(new KnnFloatVectorField("vec", vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();

                for (int i = 0; i < 10; i++) {
                    writer.deleteDocuments(new Term("id", String.valueOf(i)));
                }
                writer.forceMerge(1);
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                float[] query = vectors[0];
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery("vec", query, 10), 10);

                for (var sd : topDocs.scoreDocs) {
                    Document doc = searcher.storedFields().document(sd.doc);
                    int docId = Integer.parseInt(doc.get("id"));
                    assertTrue("Deleted doc " + docId + " should not appear in results", docId >= 10);
                }
            }
        }
    }

    public void testDeletedDocsNotInSearchResultsFlat() throws IOException {
        int dims = 8;
        int nVectors = 20;
        float[][] vectors = new float[nVectors][dims];
        Random random = new Random(42);
        for (int i = 0; i < nVectors; i++) {
            vectors[i] = randomVector(dims, random);
        }

        try (Directory dir = newDirectory()) {
            IndexWriterConfig config = new IndexWriterConfig();
            config.setCodec(getCodec(8, 8, 7, 100));
            try (IndexWriter writer = new IndexWriter(dir, config)) {
                for (int i = 0; i < nVectors; i++) {
                    Document doc = new Document();
                    doc.add(new StringField("id", String.valueOf(i), Field.Store.YES));
                    doc.add(new KnnFloatVectorField("vec", vectors[i], VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();

                for (int i = 0; i < 5; i++) {
                    writer.deleteDocuments(new Term("id", String.valueOf(i)));
                }
                writer.forceMerge(1);
                writer.commit();
            }

            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                float[] query = vectors[0];
                TopDocs topDocs = searcher.search(new KnnFloatVectorQuery("vec", query, 5), 5);

                for (var sd : topDocs.scoreDocs) {
                    Document doc = searcher.storedFields().document(sd.doc);
                    int docId = Integer.parseInt(doc.get("id"));
                    assertTrue("Deleted doc " + docId + " should not appear in results", docId >= 5);
                }
            }
        }
    }

    public void testToString() {
        IvfHnswInt8VectorsFormat format = new IvfHnswInt8VectorsFormat(256, 16, 7, 1000, 20);
        assertEquals("IvfHnswInt8VectorsFormat(nlist=256, nprobe=16, sqBits=7)", format.toString());
    }

    public void testFormatConstructorValidation() {
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(0, 16, 7, 1000, 20));
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(256, 0, 7, 1000, 20));
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(256, 16, 3, 1000, 20));
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(256, 16, 7, 0, 20));
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(256, 16, 7, 1000, 0));
        expectThrows(IllegalArgumentException.class, () -> new IvfHnswInt8VectorsFormat(4, 8, 7, 1000, 20));
    }

    private static float[] randomVector(int dims, Random random) {
        float[] vector = new float[dims];
        for (int d = 0; d < dims; d++) {
            vector[d] = random.nextFloat() * 2 - 1;
        }
        return vector;
    }
}
