/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.ingest;

import org.elasticsearch.cluster.metadata.IndexMetadata;
import org.elasticsearch.cluster.service.ClusterService;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.ingest.AbstractProcessor;
import org.elasticsearch.ingest.IngestDocument;
import org.elasticsearch.ingest.Processor;
import org.elasticsearch.plugin.encryption.EncryptionSettings;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.AesGcmCipher;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Auto-encryption processor that dynamically reads field list and key_id from
 * the target index's settings at execution time. Used by the auto-injected
 * _field_encryption default pipeline — no user configuration needed.
 */
public class AutoEncryptProcessor extends AbstractProcessor {

    public static final String TYPE = "auto_field_encrypt";
    private static final String ENCRYPTED_PREFIX = "enc:v1:";

    private final ClusterService clusterService;

    AutoEncryptProcessor(String tag, String description, ClusterService clusterService) {
        super(tag, description);
        this.clusterService = clusterService;
    }

    @Override
    public IngestDocument execute(IngestDocument document) {
        String indexName = document.getFieldValue(IngestDocument.Metadata.INDEX.getFieldName(), String.class);

        IndexMetadata indexMetadata = clusterService.state().metadata().index(indexName);
        if (indexMetadata == null) {
            return document;
        }

        Settings indexSettings = indexMetadata.getSettings();
        String keyId = EncryptionSettings.INDEX_ENCRYPTION_KEY_ID.get(indexSettings);
        List<String> fields = EncryptionSettings.INDEX_ENCRYPTION_FIELDS.get(indexSettings);

        if (keyId.isEmpty() || fields.isEmpty()) {
            return document;
        }

        byte[] aesKey = KeyServiceHolder.get().getAesKey(keyId);
        for (String field : fields) {
            if (document.hasField(field) == false) {
                continue;
            }
            Object value = document.getFieldValue(field, Object.class);
            if (value == null) {
                continue;
            }
            String plaintext = value.toString();
            if (plaintext.startsWith(ENCRYPTED_PREFIX)) {
                continue;
            }
            byte[] encrypted = AesGcmCipher.encrypt(aesKey, plaintext.getBytes(StandardCharsets.UTF_8));
            String encoded = ENCRYPTED_PREFIX + keyId + ":" + Base64.getEncoder().encodeToString(encrypted);
            document.setFieldValue(field, encoded);
        }
        return document;
    }

    @Override
    public String getType() {
        return TYPE;
    }

    public static class Factory implements Processor.Factory {

        private final ClusterService clusterService;

        public Factory(ClusterService clusterService) {
            this.clusterService = clusterService;
        }

        @Override
        public Processor create(
            Map<String, Processor.Factory> processorFactories,
            String tag,
            String description,
            Map<String, Object> config
        ) {
            return new AutoEncryptProcessor(tag, description, clusterService);
        }
    }
}
