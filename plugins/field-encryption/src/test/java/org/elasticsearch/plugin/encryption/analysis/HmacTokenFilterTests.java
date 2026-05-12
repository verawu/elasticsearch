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
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.elasticsearch.plugin.encryption.crypto.HmacCalculator;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

public class HmacTokenFilterTests extends ESTestCase {

    private byte[] hmacKey;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        hmacKey = randomByteArrayOfLength(32);
    }

    public void testSingleTokenHmac() throws IOException {
        List<String> tokens = analyzeWith("hello");

        assertEquals(1, tokens.size());
        assertEquals(HmacCalculator.hmacHex(hmacKey, "hello"), tokens.get(0));
    }

    public void testMultipleTokens() throws IOException {
        List<String> tokens = analyzeWith("the quick brown fox");

        assertEquals(4, tokens.size());
        assertEquals(HmacCalculator.hmacHex(hmacKey, "the"), tokens.get(0));
        assertEquals(HmacCalculator.hmacHex(hmacKey, "quick"), tokens.get(1));
        assertEquals(HmacCalculator.hmacHex(hmacKey, "brown"), tokens.get(2));
        assertEquals(HmacCalculator.hmacHex(hmacKey, "fox"), tokens.get(3));
    }

    public void testDeterministic() throws IOException {
        List<String> tokens1 = analyzeWith("hello world");
        List<String> tokens2 = analyzeWith("hello world");

        assertEquals(tokens1, tokens2);
    }

    public void testPositionsPreserved() throws IOException {
        MockTokenizer tokenizer = new MockTokenizer(MockTokenizer.WHITESPACE, false);
        tokenizer.setReader(new StringReader("hello world"));
        HmacTokenFilter filter = new HmacTokenFilter(tokenizer, hmacKey);

        filter.reset();
        PositionIncrementAttribute posAttr = filter.getAttribute(PositionIncrementAttribute.class);

        assertTrue(filter.incrementToken());
        assertEquals(1, posAttr.getPositionIncrement());

        assertTrue(filter.incrementToken());
        assertEquals(1, posAttr.getPositionIncrement());

        assertFalse(filter.incrementToken());
        filter.end();
        filter.close();
    }

    public void testOutputLength() throws IOException {
        List<String> tokens = analyzeWith("x");

        // HMAC-SHA256 = 32 bytes = 64 hex chars
        assertEquals(64, tokens.get(0).length());
    }

    private List<String> analyzeWith(String text) throws IOException {
        MockTokenizer tokenizer = new MockTokenizer(MockTokenizer.WHITESPACE, false);
        tokenizer.setReader(new StringReader(text));
        TokenStream stream = new HmacTokenFilter(tokenizer, hmacKey);

        List<String> result = new ArrayList<>();
        CharTermAttribute termAttr = stream.getAttribute(CharTermAttribute.class);
        stream.reset();
        while (stream.incrementToken()) {
            result.add(termAttr.toString());
        }
        stream.end();
        stream.close();
        return result;
    }
}
