/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.ingest;

import org.elasticsearch.ingest.AbstractProcessor;
import org.elasticsearch.ingest.ConfigurationUtils;
import org.elasticsearch.ingest.IngestDocument;
import org.elasticsearch.ingest.Processor;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.AesGcmCipher;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/**
 * Ingest processor that encrypts specified field values in the document source
 * using AES-256-GCM. Encrypted values are stored as: enc:v1:{key_id}:{base64(iv+ciphertext+tag)}
 */
public class EncryptProcessor extends AbstractProcessor {

    public static final String TYPE = "field_encrypt";
    private static final String ENCRYPTED_PREFIX = "enc:v1:";

    private final List<String> fields;
    private final String keyId;

    EncryptProcessor(String tag, String description, List<String> fields, String keyId) {
        super(tag, description);
        this.fields = fields;
        this.keyId = keyId;
    }

    @Override
    public IngestDocument execute(IngestDocument document) {
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
        @Override
        public Processor create(
            Map<String, Processor.Factory> processorFactories,
            String tag,
            String description,
            Map<String, Object> config
        ) {
            List<String> fields = ConfigurationUtils.readList(TYPE, tag, config, "fields");
            String keyId = ConfigurationUtils.readStringProperty(TYPE, tag, config, "key_id");
            return new EncryptProcessor(tag, description, fields, keyId);
        }
    }
}
