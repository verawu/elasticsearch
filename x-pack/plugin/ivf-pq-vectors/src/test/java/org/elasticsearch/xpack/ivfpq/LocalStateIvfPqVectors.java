/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.xpack.core.LocalStateCompositeXPackPlugin;

import java.util.List;
import java.util.Map;

public class LocalStateIvfPqVectors extends LocalStateCompositeXPackPlugin {

    private final IvfPqVectorsPlugin ivfPqPlugin;

    public LocalStateIvfPqVectors(Settings settings) {
        super(settings, null);
        ivfPqPlugin = new IvfPqVectorsPlugin();
    }

    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return ivfPqPlugin.getMappers();
    }

    @Override
    public List<SearchPlugin.QuerySpec<?>> getQueries() {
        return ivfPqPlugin.getQueries();
    }
}
