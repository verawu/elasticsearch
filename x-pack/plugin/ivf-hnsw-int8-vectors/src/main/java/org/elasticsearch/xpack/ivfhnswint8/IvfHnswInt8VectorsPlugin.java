/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfhnswint8;

import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.FieldMapper;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.plugins.MapperPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.xpack.ivfhnswint8.mapper.IvfHnswInt8KnnQueryBuilder;
import org.elasticsearch.xpack.ivfhnswint8.mapper.IvfHnswInt8VectorFieldMapper;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.index.mapper.FieldMapper.notInMultiFields;

public class IvfHnswInt8VectorsPlugin extends Plugin implements MapperPlugin, SearchPlugin {

    public static final Setting<Boolean> IVFPQ_ENABLED = Setting.boolSetting(
        "xpack.ivfhnswint8.enabled",
        true,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    private volatile boolean enabled;

    public IvfHnswInt8VectorsPlugin(Settings settings) {
        this.enabled = IVFPQ_ENABLED.get(settings);
    }

    @Override
    public Collection<?> createComponents(PluginServices services) {
        services.clusterService().getClusterSettings().addSettingsUpdateConsumer(IVFPQ_ENABLED, value -> this.enabled = value);
        return List.of();
    }

    @Override
    public Map<String, Mapper.TypeParser> getMappers() {
        return Map.of(
            IvfHnswInt8VectorFieldMapper.CONTENT_TYPE,
            new FieldMapper.TypeParser(
                (n, c) -> {
                    if (enabled == false) {
                        throw new IllegalArgumentException(
                            "IVF-PQ vectors are disabled. Set [xpack.ivfhnswint8.enabled] to [true] to enable."
                        );
                    }
                    return new IvfHnswInt8VectorFieldMapper.Builder(n);
                },
                notInMultiFields(IvfHnswInt8VectorFieldMapper.CONTENT_TYPE)
            )
        );
    }

    @Override
    public List<QuerySpec<?>> getQueries() {
        return List.of(
            new QuerySpec<>(IvfHnswInt8KnnQueryBuilder.NAME, IvfHnswInt8KnnQueryBuilder::new, IvfHnswInt8KnnQueryBuilder::fromXContent)
        );
    }

    @Override
    public List<Setting<?>> getSettings() {
        return List.of(IVFPQ_ENABLED);
    }
}
