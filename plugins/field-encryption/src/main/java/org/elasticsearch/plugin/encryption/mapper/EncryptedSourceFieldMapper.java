/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.mapper;

import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.IndexableField;
import org.elasticsearch.common.bytes.BytesReference;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.mapper.DocumentParserContext;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.mapper.MetadataFieldMapper;
import org.elasticsearch.index.mapper.TextSearchInfo;
import org.elasticsearch.index.mapper.ValueFetcher;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugin.encryption.EncryptionSettings;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.AesGcmCipher;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;
import org.elasticsearch.xcontent.XContentParserConfiguration;
import org.elasticsearch.xcontent.XContentType;
import org.elasticsearch.xcontent.json.JsonXContent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A metadata field mapper that encrypts configured fields in _source AFTER analysis.
 * This runs in postParse(), which executes after all field analyzers have processed
 * the plaintext — so HMAC token filters see the original values, while the stored
 * _source contains encrypted ciphertext.
 */
public class EncryptedSourceFieldMapper extends MetadataFieldMapper {

    public static final String NAME = "_encrypted_source";
    private static final String ENCRYPTED_PREFIX = "enc:v1:";

    public static final TypeParser PARSER = new ConfigurableTypeParser(
        c -> new EncryptedSourceFieldMapper(),
        c -> new Builder()
    );

    private static class Builder extends MetadataFieldMapper.Builder {
        Builder() {
            super(NAME);
        }

        @Override
        protected Parameter<?>[] getParameters() {
            return new Parameter<?>[0];
        }

        @Override
        public MetadataFieldMapper build() {
            return new EncryptedSourceFieldMapper();
        }
    }

    private EncryptedSourceFieldMapper() {
        super(new EncryptedSourceFieldType());
    }

    private static class EncryptedSourceFieldType extends MappedFieldType {
        EncryptedSourceFieldType() {
            super(NAME, false, false, false, TextSearchInfo.NONE, Collections.emptyMap());
        }

        @Override
        public String typeName() {
            return NAME;
        }

        @Override
        public ValueFetcher valueFetcher(SearchExecutionContext context, String format) {
            return ValueFetcher.EMPTY;
        }

        @Override
        public org.apache.lucene.search.Query termQuery(Object value, SearchExecutionContext context) {
            throw new UnsupportedOperationException("_encrypted_source field is not searchable");
        }
    }

    @Override
    public void postParse(DocumentParserContext context) throws IOException {
        IndexSettings indexSettings = context.indexSettings();
        String keyId = EncryptionSettings.INDEX_ENCRYPTION_KEY_ID.get(indexSettings.getSettings());
        List<String> fields = EncryptionSettings.INDEX_ENCRYPTION_FIELDS.get(indexSettings.getSettings());

        if (keyId.isEmpty() || fields.isEmpty()) {
            return;
        }

        // Find the _source stored field
        IndexableField sourceField = context.doc().getField("_source");
        if (sourceField == null) {
            return;
        }

        byte[] sourceBytes = sourceField.binaryValue().bytes;
        int offset = sourceField.binaryValue().offset;
        int length = sourceField.binaryValue().length;

        // Parse the source JSON
        Map<String, Object> sourceMap;
        try (
            XContentParser parser = XContentType.JSON.xContent()
                .createParser(XContentParserConfiguration.EMPTY, sourceBytes, offset, length)
        ) {
            sourceMap = parser.map();
        }

        // Encrypt configured fields
        byte[] aesKey = KeyServiceHolder.get().getAesKey(keyId);
        boolean modified = encryptFields(sourceMap, fields, aesKey, keyId);

        if (modified == false) {
            return;
        }

        // Rebuild source bytes with encrypted fields
        BytesReference encryptedSource;
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            builder.map(sourceMap);
            encryptedSource = BytesReference.bytes(builder);
        }

        // Replace _source in the document
        context.doc().getFields().removeIf(f -> "_source".equals(f.name()));
        byte[] encBytes = BytesReference.toBytes(encryptedSource);
        context.doc().add(new StoredField("_source", encBytes, 0, encBytes.length));
    }

    private boolean encryptFields(Map<String, Object> sourceMap, List<String> fields, byte[] aesKey, String keyId) {
        boolean modified = false;
        for (String field : fields) {
            if (field.contains(".")) {
                modified |= encryptNestedField(sourceMap, field.split("\\."), 0, aesKey, keyId);
            } else {
                Object value = sourceMap.get(field);
                if (value != null && value instanceof String strValue) {
                    if (strValue.startsWith(ENCRYPTED_PREFIX) == false) {
                        byte[] encrypted = AesGcmCipher.encrypt(aesKey, strValue.getBytes(StandardCharsets.UTF_8));
                        sourceMap.put(field, ENCRYPTED_PREFIX + keyId + ":" + Base64.getEncoder().encodeToString(encrypted));
                        modified = true;
                    }
                }
            }
        }
        return modified;
    }

    @SuppressWarnings("unchecked")
    private boolean encryptNestedField(Map<String, Object> map, String[] path, int idx, byte[] aesKey, String keyId) {
        if (idx >= path.length) return false;
        String key = path[idx];
        Object value = map.get(key);
        if (value == null) return false;

        if (idx == path.length - 1) {
            if (value instanceof String strValue && strValue.startsWith(ENCRYPTED_PREFIX) == false) {
                byte[] encrypted = AesGcmCipher.encrypt(aesKey, strValue.getBytes(StandardCharsets.UTF_8));
                map.put(key, ENCRYPTED_PREFIX + keyId + ":" + Base64.getEncoder().encodeToString(encrypted));
                return true;
            }
            return false;
        }

        if (value instanceof Map) {
            return encryptNestedField((Map<String, Object>) value, path, idx + 1, aesKey, keyId);
        }
        return false;
    }

    @Override
    protected String contentType() {
        return NAME;
    }
}
