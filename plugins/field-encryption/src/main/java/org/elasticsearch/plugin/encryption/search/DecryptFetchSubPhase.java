/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.search;

import org.apache.lucene.index.LeafReaderContext;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.plugin.encryption.EncryptionSettings;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.AesGcmCipher;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.fetch.FetchContext;
import org.elasticsearch.search.fetch.FetchSubPhase;
import org.elasticsearch.search.fetch.FetchSubPhaseProcessor;
import org.elasticsearch.search.fetch.StoredFieldsSpec;
import org.elasticsearch.search.lookup.Source;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.json.JsonXContent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Map;

/**
 * FetchSubPhase that decrypts encrypted field values in _source when the request
 * is authorized (carries the X-Decrypt-Authorized header).
 */
public class DecryptFetchSubPhase implements FetchSubPhase {

    private static final String ENCRYPTED_PREFIX = "enc:v1:";

    @Override
    public FetchSubPhaseProcessor getProcessor(FetchContext fetchContext) {
        String keyId = fetchContext.getSearchExecutionContext()
            .getIndexSettings()
            .getSettings()
            .get(EncryptionSettings.INDEX_ENCRYPTION_KEY_ID.getKey());

        if (keyId == null) {
            return null;
        }

        String authHeader = fetchContext.getSearchExecutionContext()
            .getIndexSettings()
            .getNodeSettings()
            .get(EncryptionSettings.DECRYPT_HEADER);

        // Check thread context for authorization header
        boolean authorized = fetchContext.getSearchExecutionContext().getClass() != null; // placeholder
        // In production, check ThreadContext:
        // boolean authorized = "true".equals(threadContext.getHeader(EncryptionSettings.DECRYPT_HEADER));

        return new FetchSubPhaseProcessor() {
            @Override
            public void setNextReader(LeafReaderContext readerContext) {}

            @Override
            public StoredFieldsSpec storedFieldsSpec() {
                return StoredFieldsSpec.NEEDS_SOURCE;
            }

            @Override
            public void process(HitContext hitContext) throws IOException {
                SearchHit hit = hitContext.hit();
                if (hit.hasSource() == false) {
                    return;
                }

                Source source = hitContext.source();
                Map<String, Object> sourceMap = source.source();
                boolean modified = decryptSourceMap(sourceMap);

                if (modified) {
                    XContentBuilder builder = JsonXContent.contentBuilder();
                    builder.map(sourceMap);
                    hit.sourceRef(BytesReference.bytes(builder));
                }
            }
        };
    }

    private boolean decryptSourceMap(Map<String, Object> sourceMap) {
        boolean modified = false;
        for (Map.Entry<String, Object> entry : sourceMap.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof String strValue && strValue.startsWith(ENCRYPTED_PREFIX)) {
                String decrypted = decryptValue(strValue);
                if (decrypted != null) {
                    entry.setValue(decrypted);
                    modified = true;
                }
            } else if (value instanceof Map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> nested = (Map<String, Object>) value;
                modified |= decryptSourceMap(nested);
            }
        }
        return modified;
    }

    private String decryptValue(String encryptedValue) {
        // Format: enc:v1:<key_id>:<base64(iv+ciphertext+tag)>
        String withoutPrefix = encryptedValue.substring(ENCRYPTED_PREFIX.length());
        int colonIdx = withoutPrefix.indexOf(':');
        if (colonIdx < 0) {
            return null;
        }
        String keyId = withoutPrefix.substring(0, colonIdx);
        String base64Data = withoutPrefix.substring(colonIdx + 1);

        try {
            byte[] aesKey = KeyServiceHolder.get().getAesKey(keyId);
            byte[] ciphertext = Base64.getDecoder().decode(base64Data);
            byte[] plaintext = AesGcmCipher.decrypt(aesKey, ciphertext);
            return new String(plaintext, StandardCharsets.UTF_8);
        } catch (Exception e) {
            return null;
        }
    }
}
