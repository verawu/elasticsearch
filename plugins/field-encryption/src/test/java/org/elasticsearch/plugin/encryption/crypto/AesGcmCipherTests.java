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

import java.nio.charset.StandardCharsets;

public class AesGcmCipherTests extends ESTestCase {

    public void testEncryptDecryptRoundtrip() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = "hello world".getBytes(StandardCharsets.UTF_8);

        byte[] ciphertext = AesGcmCipher.encrypt(key, plaintext);
        byte[] decrypted = AesGcmCipher.decrypt(key, ciphertext);

        assertEquals("hello world", new String(decrypted, StandardCharsets.UTF_8));
    }

    public void testEncryptDecryptEmptyPlaintext() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = new byte[0];

        byte[] ciphertext = AesGcmCipher.encrypt(key, plaintext);
        byte[] decrypted = AesGcmCipher.decrypt(key, ciphertext);

        assertEquals(0, decrypted.length);
    }

    public void testEncryptDecryptLongPlaintext() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = randomByteArrayOfLength(10000);

        byte[] ciphertext = AesGcmCipher.encrypt(key, plaintext);
        byte[] decrypted = AesGcmCipher.decrypt(key, ciphertext);

        assertArrayEquals(plaintext, decrypted);
    }

    public void testNonDeterministic() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = "same input".getBytes(StandardCharsets.UTF_8);

        byte[] ciphertext1 = AesGcmCipher.encrypt(key, plaintext);
        byte[] ciphertext2 = AesGcmCipher.encrypt(key, plaintext);

        // Different random IVs should produce different ciphertexts
        assertFalse(java.util.Arrays.equals(ciphertext1, ciphertext2));

        // But both should decrypt to the same plaintext
        assertEquals(
            new String(AesGcmCipher.decrypt(key, ciphertext1), StandardCharsets.UTF_8),
            new String(AesGcmCipher.decrypt(key, ciphertext2), StandardCharsets.UTF_8)
        );
    }

    public void testCiphertextFormat() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = "test".getBytes(StandardCharsets.UTF_8);

        byte[] ciphertext = AesGcmCipher.encrypt(key, plaintext);

        // IV (12) + plaintext (4) + GCM tag (16) = 32
        assertEquals(12 + 4 + 16, ciphertext.length);
    }

    public void testWrongKeyFails() {
        byte[] key1 = randomByteArrayOfLength(32);
        byte[] key2 = randomByteArrayOfLength(32);
        byte[] plaintext = "secret".getBytes(StandardCharsets.UTF_8);

        byte[] ciphertext = AesGcmCipher.encrypt(key1, plaintext);

        expectThrows(IllegalStateException.class, () -> AesGcmCipher.decrypt(key2, ciphertext));
    }

    public void testTamperedCiphertextFails() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] plaintext = "secret".getBytes(StandardCharsets.UTF_8);

        byte[] ciphertext = AesGcmCipher.encrypt(key, plaintext);
        // Flip a bit in the ciphertext body
        ciphertext[15] ^= 0x01;

        expectThrows(IllegalStateException.class, () -> AesGcmCipher.decrypt(key, ciphertext));
    }

    public void testTooShortCiphertextFails() {
        byte[] key = randomByteArrayOfLength(32);
        byte[] tooShort = new byte[10];

        expectThrows(IllegalArgumentException.class, () -> AesGcmCipher.decrypt(key, tooShort));
    }
}
