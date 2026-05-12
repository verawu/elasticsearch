/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.crypto;

import org.elasticsearch.test.ESTestCase;

import java.util.HexFormat;

/**
 * Tests for HKDF-SHA256 using RFC 5869 test vectors.
 */
public class HkdfDeriverTests extends ESTestCase {

    private static final HexFormat HEX = HexFormat.of();

    // RFC 5869 Test Case 1
    public void testRfc5869TestCase1() {
        byte[] ikm = HEX.parseHex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        byte[] salt = HEX.parseHex("000102030405060708090a0b0c");
        byte[] info = HEX.parseHex("f0f1f2f3f4f5f6f7f8f9");
        int length = 42;

        byte[] okm = HkdfDeriver.derive(ikm, salt, info, length);

        String expected = "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865";
        assertEquals(expected, HEX.formatHex(okm));
    }

    // RFC 5869 Test Case 2
    public void testRfc5869TestCase2() {
        byte[] ikm = HEX.parseHex(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
                + "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
                + "404142434445464748494a4b4c4d4e4f"
        );
        byte[] salt = HEX.parseHex(
            "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f"
                + "808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f"
                + "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
        );
        byte[] info = HEX.parseHex(
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
                + "d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeef"
                + "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
        );
        int length = 82;

        byte[] okm = HkdfDeriver.derive(ikm, salt, info, length);

        String expected = "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c"
            + "59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71"
            + "cc30c58179ec3e87c14c01d5c1f3434f1d87";
        assertEquals(expected, HEX.formatHex(okm));
    }

    // RFC 5869 Test Case 3 (empty salt and info)
    public void testRfc5869TestCase3() {
        byte[] ikm = HEX.parseHex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        byte[] salt = new byte[0];
        byte[] info = new byte[0];
        int length = 42;

        byte[] okm = HkdfDeriver.derive(ikm, salt, info, length);

        String expected = "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8";
        assertEquals(expected, HEX.formatHex(okm));
    }

    public void testDeterministic() {
        byte[] ikm = randomByteArrayOfLength(32);
        byte[] info = "test-info".getBytes();
        byte[] result1 = HkdfDeriver.derive(ikm, null, info, 32);
        byte[] result2 = HkdfDeriver.derive(ikm, null, info, 32);
        assertArrayEquals(result1, result2);
    }

    public void testDifferentInfoProducesDifferentKeys() {
        byte[] ikm = randomByteArrayOfLength(32);
        byte[] result1 = HkdfDeriver.derive(ikm, null, "source-encryption:key1".getBytes(), 32);
        byte[] result2 = HkdfDeriver.derive(ikm, null, "term-encryption:key1".getBytes(), 32);
        assertNotEquals(HEX.formatHex(result1), HEX.formatHex(result2));
    }

    public void testInvalidLength() {
        byte[] ikm = randomByteArrayOfLength(32);
        expectThrows(IllegalArgumentException.class, () -> HkdfDeriver.derive(ikm, null, new byte[0], 0));
        expectThrows(IllegalArgumentException.class, () -> HkdfDeriver.derive(ikm, null, new byte[0], 256 * 32));
    }
}
