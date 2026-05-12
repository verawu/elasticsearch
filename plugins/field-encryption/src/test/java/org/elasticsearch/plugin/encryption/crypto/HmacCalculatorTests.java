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

public class HmacCalculatorTests extends ESTestCase {

    // RFC 4231 Test Case 2
    public void testRfc4231TestCase2() {
        byte[] key = "Jefe".getBytes();
        byte[] data = "what do ya want for nothing?".getBytes();

        byte[] result = HmacCalculator.hmac(key, data);

        String expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
        assertEquals(expected, HexFormat.of().formatHex(result));
    }

    public void testHmacHexDeterministic() {
        byte[] key = randomByteArrayOfLength(32);
        String input = "alice@example.com";

        String result1 = HmacCalculator.hmacHex(key, input);
        String result2 = HmacCalculator.hmacHex(key, input);

        assertEquals(result1, result2);
    }

    public void testHmacHexLength() {
        byte[] key = randomByteArrayOfLength(32);
        String result = HmacCalculator.hmacHex(key, "test");

        // HMAC-SHA256 produces 32 bytes = 64 hex chars
        assertEquals(64, result.length());
    }

    public void testDifferentInputsDifferentOutputs() {
        byte[] key = randomByteArrayOfLength(32);

        String result1 = HmacCalculator.hmacHex(key, "alice@example.com");
        String result2 = HmacCalculator.hmacHex(key, "bob@example.com");

        assertNotEquals(result1, result2);
    }

    public void testDifferentKeysDifferentOutputs() {
        byte[] key1 = randomByteArrayOfLength(32);
        byte[] key2 = randomByteArrayOfLength(32);

        String result1 = HmacCalculator.hmacHex(key1, "same-input");
        String result2 = HmacCalculator.hmacHex(key2, "same-input");

        assertNotEquals(result1, result2);
    }

    public void testHexOutputIsLowercase() {
        byte[] key = randomByteArrayOfLength(32);
        String result = HmacCalculator.hmacHex(key, "test");

        assertEquals(result, result.toLowerCase());
    }
}
