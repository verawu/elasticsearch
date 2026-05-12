/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.analysis;

import org.apache.lucene.analysis.TokenStream;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.analysis.AbstractTokenFilterFactory;
import org.elasticsearch.plugin.encryption.EncryptionSettings;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;

/**
 * Factory for the hmac_encrypt token filter.
 * Reads key_id from filter settings (falling back to index-level encryption.key_id),
 * resolves the HMAC key via KeyService, and creates HmacTokenFilter instances.
 */
public class HmacTokenFilterFactory extends AbstractTokenFilterFactory {

    private final byte[] hmacKey;

    public HmacTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
        super(name, settings);
        String keyId = settings.get("key_id");
        if (keyId == null) {
            keyId = indexSettings.getSettings().get(EncryptionSettings.INDEX_ENCRYPTION_KEY_ID.getKey());
        }
        if (keyId == null) {
            throw new IllegalArgumentException(
                "hmac_encrypt token filter requires 'key_id' setting or index-level 'index.encryption.key_id'"
            );
        }
        this.hmacKey = KeyServiceHolder.get().getHmacKey(keyId);
    }

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new HmacTokenFilter(tokenStream, hmacKey);
    }
}
