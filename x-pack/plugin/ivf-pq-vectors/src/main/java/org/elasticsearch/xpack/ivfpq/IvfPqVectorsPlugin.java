/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq;

import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.plugins.MapperPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.xpack.ivfpq.mapper.IvfPqVectorFieldMapper;

import java.util.Map;

import static org.elasticsearch.index.mapper.FieldMapper.notInMultiFields;

public class IvfPqVectorsPlugin extends Plugin implements MapperPlugin {

    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return Map.of(
            IvfPqVectorFieldMapper.CONTENT_TYPE,
            new Mapper.TypeParser((n, c) -> new IvfPqVectorFieldMapper.Builder(n), notInMultiFields(IvfPqVectorFieldMapper.CONTENT_TYPE))
        );
    }
}
