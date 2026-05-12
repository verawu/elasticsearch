/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.crypto;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;

import java.security.GeneralSecurityException;
import java.security.SecureRandom;

/**
 * AES-256-GCM encryption and decryption.
 * Wire format: IV (12 bytes) || ciphertext || GCM auth tag (16 bytes appended by cipher).
 */
public final class AesGcmCipher {

    private static final String ALGO = "AES/GCM/NoPadding";
    private static final int IV_LENGTH = 12;
    private static final int TAG_BIT_LENGTH = 128;
    private static final SecureRandom RANDOM = new SecureRandom();

    private AesGcmCipher() {}

    public static byte[] encrypt(byte[] key, byte[] plaintext) {
        try {
            byte[] iv = new byte[IV_LENGTH];
            RANDOM.nextBytes(iv);

            Cipher cipher = Cipher.getInstance(ALGO);
            cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"), new GCMParameterSpec(TAG_BIT_LENGTH, iv));
            byte[] ciphertextWithTag = cipher.doFinal(plaintext);

            byte[] output = new byte[IV_LENGTH + ciphertextWithTag.length];
            System.arraycopy(iv, 0, output, 0, IV_LENGTH);
            System.arraycopy(ciphertextWithTag, 0, output, IV_LENGTH, ciphertextWithTag.length);
            return output;
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("AES-GCM encryption failed", e);
        }
    }

    public static byte[] decrypt(byte[] key, byte[] ciphertext) {
        if (ciphertext.length < IV_LENGTH + TAG_BIT_LENGTH / 8) {
            throw new IllegalArgumentException("Ciphertext too short");
        }
        try {
            Cipher cipher = Cipher.getInstance(ALGO);
            GCMParameterSpec spec = new GCMParameterSpec(TAG_BIT_LENGTH, ciphertext, 0, IV_LENGTH);
            cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"), spec);
            return cipher.doFinal(ciphertext, IV_LENGTH, ciphertext.length - IV_LENGTH);
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("AES-GCM decryption failed", e);
        }
    }
}
