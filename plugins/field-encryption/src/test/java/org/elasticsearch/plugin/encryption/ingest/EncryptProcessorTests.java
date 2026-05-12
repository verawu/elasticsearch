/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.ingest;

import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.ingest.IngestDocument;
import org.elasticsearch.ingest.TestIngestDocument;
import org.elasticsearch.plugin.encryption.KeyService;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.AesGcmCipher;
import org.elasticsearch.test.ESTestCase;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EncryptProcessorTests extends ESTestCase {

    private static final String KEY_ID = "test-key";

    @Override
    public void setUp() throws Exception {
        super.setUp();
        String masterKeyB64 = Base64.getEncoder().encodeToString(randomByteArrayOfLength(32));
        Settings settings = Settings.builder().put("encryption.master_key." + KEY_ID, masterKeyB64).build();
        KeyServiceHolder.set(new KeyService(settings));
    }

    @Override
    public void tearDown() throws Exception {
        KeyServiceHolder.clear();
        super.tearDown();
    }

    public void testEncryptsSingleField() throws Exception {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email"), KEY_ID);

        Map<String, Object> source = new HashMap<>();
        source.put("email", "alice@example.com");
        source.put("name", "Alice");
        IngestDocument doc = TestIngestDocument.withDefaultVersion(source);

        processor.execute(doc);

        String encrypted = doc.getFieldValue("email", String.class);
        assertTrue(encrypted.startsWith("enc:v1:" + KEY_ID + ":"));
        // name should not be affected
        assertEquals("Alice", doc.getFieldValue("name", String.class));

        // Verify decryption works
        String base64Part = encrypted.substring(("enc:v1:" + KEY_ID + ":").length());
        byte[] ciphertext = Base64.getDecoder().decode(base64Part);
        byte[] aesKey = KeyServiceHolder.get().getAesKey(KEY_ID);
        byte[] plaintext = AesGcmCipher.decrypt(aesKey, ciphertext);
        assertEquals("alice@example.com", new String(plaintext, StandardCharsets.UTF_8));
    }

    public void testEncryptsMultipleFields() throws Exception {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email", "phone"), KEY_ID);

        Map<String, Object> source = new HashMap<>();
        source.put("email", "alice@example.com");
        source.put("phone", "+1234567890");
        IngestDocument doc = TestIngestDocument.withDefaultVersion(source);

        processor.execute(doc);

        assertTrue(doc.getFieldValue("email", String.class).startsWith("enc:v1:"));
        assertTrue(doc.getFieldValue("phone", String.class).startsWith("enc:v1:"));
    }

    public void testSkipsMissingFields() throws Exception {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email", "missing"), KEY_ID);

        Map<String, Object> source = new HashMap<>();
        source.put("email", "alice@example.com");
        IngestDocument doc = TestIngestDocument.withDefaultVersion(source);

        // Should not throw
        processor.execute(doc);

        assertTrue(doc.getFieldValue("email", String.class).startsWith("enc:v1:"));
    }

    public void testSkipsNullFields() throws Exception {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email"), KEY_ID);

        Map<String, Object> source = new HashMap<>();
        source.put("email", null);
        IngestDocument doc = TestIngestDocument.withDefaultVersion(source);

        processor.execute(doc);

        assertNull(doc.getFieldValue("email", Object.class));
    }

    public void testNonDeterministicEncryption() throws Exception {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email"), KEY_ID);

        Map<String, Object> source1 = new HashMap<>();
        source1.put("email", "alice@example.com");
        IngestDocument doc1 = TestIngestDocument.withDefaultVersion(source1);
        processor.execute(doc1);

        Map<String, Object> source2 = new HashMap<>();
        source2.put("email", "alice@example.com");
        IngestDocument doc2 = TestIngestDocument.withDefaultVersion(source2);
        processor.execute(doc2);

        // Same plaintext should produce different ciphertexts (random IV)
        assertNotEquals(doc1.getFieldValue("email", String.class), doc2.getFieldValue("email", String.class));
    }

    public void testGetType() {
        EncryptProcessor processor = new EncryptProcessor("tag", "desc", List.of("email"), KEY_ID);
        assertEquals("field_encrypt", processor.getType());
    }
}
