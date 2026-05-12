/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugin.encryption.crypto.HkdfDeriver;

import java.io.Closeable;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.concurrent.ConcurrentHashMap;

public class KeyService implements Closeable {

    private final Settings settings;
    private final ConcurrentHashMap<String, DerivedKeys> cache = new ConcurrentHashMap<>();

    public KeyService(Settings settings) {
        this.settings = settings;
    }

    public byte[] getAesKey(String keyId) {
        return getDerivedKeys(keyId).aesKey;
    }

    public byte[] getHmacKey(String keyId) {
        return getDerivedKeys(keyId).hmacKey;
    }

    private DerivedKeys getDerivedKeys(String keyId) {
        return cache.computeIfAbsent(keyId, this::deriveKeys);
    }

    private DerivedKeys deriveKeys(String keyId) {
        byte[] masterKey = loadMasterKey(keyId);
        byte[] aesKey = HkdfDeriver.derive(masterKey, null, ("source-encryption:" + keyId).getBytes(StandardCharsets.UTF_8), 32);
        byte[] hmacKey = HkdfDeriver.derive(masterKey, null, ("term-encryption:" + keyId).getBytes(StandardCharsets.UTF_8), 32);
        Arrays.fill(masterKey, (byte) 0);
        return new DerivedKeys(aesKey, hmacKey);
    }

    private byte[] loadMasterKey(String keyId) {
        String settingKey = EncryptionSettings.KEY_PREFIX + keyId;
        String value = settings.get(settingKey);
        if (value == null) {
            throw new IllegalArgumentException(
                "Encryption master key not found for key_id [" + keyId + "]. "
                    + "Add it with: elasticsearch-keystore add " + settingKey
            );
        }
        return Base64.getDecoder().decode(value);
    }

    @Override
    public void close() {
        cache.values().forEach(DerivedKeys::clear);
        cache.clear();
    }

    private static class DerivedKeys {
        final byte[] aesKey;
        final byte[] hmacKey;

        DerivedKeys(byte[] aesKey, byte[] hmacKey) {
            this.aesKey = aesKey;
            this.hmacKey = hmacKey;
        }

        void clear() {
            Arrays.fill(aesKey, (byte) 0);
            Arrays.fill(hmacKey, (byte) 0);
        }
    }
}
