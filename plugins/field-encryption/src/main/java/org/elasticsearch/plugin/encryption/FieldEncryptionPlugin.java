/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption;

import org.elasticsearch.common.settings.Setting;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.index.analysis.TokenFilterFactory;
import org.elasticsearch.index.mapper.MetadataFieldMapper;
import org.elasticsearch.indices.analysis.AnalysisModule;
import org.elasticsearch.plugin.encryption.analysis.HmacTokenFilterFactory;
import org.elasticsearch.plugin.encryption.mapper.EncryptedSourceFieldMapper;
import org.elasticsearch.plugin.encryption.search.DecryptFetchSubPhase;
import org.elasticsearch.plugin.encryption.search.EncryptedTermQueryBuilder;
import org.elasticsearch.plugins.AnalysisPlugin;
import org.elasticsearch.plugins.MapperPlugin;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.SearchPlugin;
import org.elasticsearch.search.fetch.FetchSubPhase;

import java.util.Collection;
import java.util.List;
import java.util.Map;

public class FieldEncryptionPlugin extends Plugin implements AnalysisPlugin, MapperPlugin, SearchPlugin {

    private final Settings settings;

    public FieldEncryptionPlugin(Settings settings) {
        this.settings = settings;
    }

    @Override
    public Collection<?> createComponents(PluginServices services) {
        KeyService keyService = new KeyService(services.environment().settings());
        KeyServiceHolder.set(keyService);
        return List.of(keyService);
    }

    @Override
    public List<Setting<?>> getSettings() {
        return EncryptionSettings.getSettings();
    }

    @Override
    public Map<String, AnalysisModule.AnalysisProvider<TokenFilterFactory>> getTokenFilters() {
        return Map.of("hmac_encrypt", HmacTokenFilterFactory::new);
    }

    @Override
    public Map<String, MetadataFieldMapper.TypeParser> getMetadataMappers() {
        return Map.of(EncryptedSourceFieldMapper.NAME, EncryptedSourceFieldMapper.PARSER);
    }

    @Override
    public List<FetchSubPhase> getFetchSubPhases(FetchPhaseConstructionContext context) {
        return List.of(new DecryptFetchSubPhase());
    }

    @Override
    public List<QuerySpec<?>> getQueries() {
        return List.of(
            new QuerySpec<>(EncryptedTermQueryBuilder.NAME, EncryptedTermQueryBuilder::new, EncryptedTermQueryBuilder::fromXContent)
        );
    }

    @Override
    public void close() {
        KeyServiceHolder.clear();
    }
}
