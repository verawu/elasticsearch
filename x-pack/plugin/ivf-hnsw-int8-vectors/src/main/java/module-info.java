/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

module org.elasticsearch.ivfhnswint8.vectors {
    requires org.elasticsearch.xcore;
    requires org.elasticsearch.server;
    requires org.apache.lucene.core;
    requires org.elasticsearch.xcontent;
    requires org.apache.logging.log4j;

    exports org.elasticsearch.xpack.ivfhnswint8;
    exports org.elasticsearch.xpack.ivfhnswint8.mapper;
    exports org.elasticsearch.xpack.ivfhnswint8.codec;
    exports org.elasticsearch.xpack.ivfhnswint8.training;

    provides org.elasticsearch.features.FeatureSpecification
        with org.elasticsearch.xpack.ivfhnswint8.IvfHnswInt8Features;
    provides org.apache.lucene.codecs.KnnVectorsFormat
        with org.elasticsearch.xpack.ivfhnswint8.codec.IvfHnswInt8VectorsFormat;
}
