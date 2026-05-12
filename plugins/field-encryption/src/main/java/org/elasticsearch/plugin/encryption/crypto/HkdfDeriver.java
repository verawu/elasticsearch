/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.crypto;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import java.security.GeneralSecurityException;
import java.util.Arrays;

/**
 * HKDF-SHA256 key derivation (RFC 5869).
 */
public final class HkdfDeriver {

    private static final String HMAC_ALGO = "HmacSHA256";
    private static final int HASH_LEN = 32;

    private HkdfDeriver() {}

    public static byte[] derive(byte[] ikm, byte[] salt, byte[] info, int length) {
        if (length <= 0 || length > 255 * HASH_LEN) {
            throw new IllegalArgumentException("Invalid output length: " + length);
        }
        byte[] prk = extract(salt, ikm);
        return expand(prk, info, length);
    }

    private static byte[] extract(byte[] salt, byte[] ikm) {
        if (salt == null || salt.length == 0) {
            salt = new byte[HASH_LEN];
        }
        return hmacSha256(salt, ikm);
    }

    private static byte[] expand(byte[] prk, byte[] info, int length) {
        int n = (length + HASH_LEN - 1) / HASH_LEN;
        byte[] output = new byte[n * HASH_LEN];
        byte[] t = new byte[0];

        for (int i = 1; i <= n; i++) {
            byte[] input = new byte[t.length + info.length + 1];
            System.arraycopy(t, 0, input, 0, t.length);
            System.arraycopy(info, 0, input, t.length, info.length);
            input[input.length - 1] = (byte) i;
            t = hmacSha256(prk, input);
            System.arraycopy(t, 0, output, (i - 1) * HASH_LEN, HASH_LEN);
        }

        return Arrays.copyOf(output, length);
    }

    private static byte[] hmacSha256(byte[] key, byte[] data) {
        try {
            Mac mac = Mac.getInstance(HMAC_ALGO);
            mac.init(new SecretKeySpec(key, HMAC_ALGO));
            return mac.doFinal(data);
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("HMAC-SHA256 not available", e);
        }
    }
}
