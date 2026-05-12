/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption;

import org.elasticsearch.cluster.metadata.Metadata;
import org.elasticsearch.common.compress.CompressedXContent;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.index.IndexMode;
import org.elasticsearch.index.IndexSettingProvider;
import org.elasticsearch.index.IndexSettings;

import java.time.Instant;
import java.util.List;

/**
 * Automatically injects index.default_pipeline when encryption fields are configured.
 * This eliminates the need for users to manually create or reference an ingest pipeline.
 */
public class EncryptionIndexSettingProvider implements IndexSettingProvider {

    @Override
    public Settings getAdditionalIndexSettings(
        String indexName,
        @Nullable String dataStreamName,
        @Nullable IndexMode templateIndexMode,
        Metadata metadata,
        Instant resolvedAt,
        Settings indexTemplateAndCreateRequestSettings,
        List<CompressedXContent> combinedTemplateMappings
    ) {
        String keyId = EncryptionSettings.INDEX_ENCRYPTION_KEY_ID.get(indexTemplateAndCreateRequestSettings);
        List<String> fields = EncryptionSettings.INDEX_ENCRYPTION_FIELDS.get(indexTemplateAndCreateRequestSettings);

        if (keyId.isEmpty() || fields.isEmpty()) {
            return Settings.EMPTY;
        }

        // Only inject if no pipeline is already configured
        String existingPipeline = IndexSettings.DEFAULT_PIPELINE.get(indexTemplateAndCreateRequestSettings);
        if ("_none".equals(existingPipeline) == false && existingPipeline.isEmpty() == false) {
            return Settings.EMPTY;
        }

        return Settings.builder().put(IndexSettings.DEFAULT_PIPELINE.getKey(), EncryptionSettings.PIPELINE_NAME).build();
    }
}
