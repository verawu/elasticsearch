/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq;

import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.mapper.FieldMapper;
import org.elasticsearch.index.mapper.Mapper;
import org.elasticsearch.plugins.MapperPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.xpack.ivfpq.mapper.IvfPqKnnQueryBuilder;
import org.elasticsearch.xpack.ivfpq.mapper.IvfPqVectorFieldMapper;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.elasticsearch.index.mapper.FieldMapper.notInMultiFields;

public class IvfPqVectorsPlugin extends Plugin implements MapperPlugin, SearchPlugin {

    public static final Setting<Boolean> IVFPQ_ENABLED = Setting.boolSetting(
        "xpack.ivfpq.enabled",
        true,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    private volatile boolean enabled;

    public IvfPqVectorsPlugin(Settings settings) {
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
            IvfPqVectorFieldMapper.CONTENT_TYPE,
            new FieldMapper.TypeParser(
                (n, c) -> {
                    if (enabled == false) {
                        throw new IllegalArgumentException(
                            "IVF-PQ vectors are disabled. Set [xpack.ivfpq.enabled] to [true] to enable."
                        );
                    }
                    return new IvfPqVectorFieldMapper.Builder(n);
                },
                notInMultiFields(IvfPqVectorFieldMapper.CONTENT_TYPE)
            )
        );
    }

    @Override
    public List<QuerySpec<?>> getQueries() {
        return List.of(
            new QuerySpec<>(IvfPqKnnQueryBuilder.NAME, IvfPqKnnQueryBuilder::new, IvfPqKnnQueryBuilder::fromXContent)
        );
    }

    @Override
    public List<Setting<?>> getSettings() {
        return List.of(IVFPQ_ENABLED);
    }
}
