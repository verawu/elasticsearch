/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.analysis;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.elasticsearch.plugin.encryption.crypto.HmacCalculator;

import java.io.IOException;

/**
 * A TokenFilter that replaces each token's text with its HMAC-SHA256 hex digest.
 * Positions and offsets are preserved — only the term text changes.
 */
public final class HmacTokenFilter extends TokenFilter {

    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);
    private final byte[] hmacKey;

    public HmacTokenFilter(TokenStream input, byte[] hmacKey) {
        super(input);
        this.hmacKey = hmacKey;
    }

    @Override
    public boolean incrementToken() throws IOException {
        if (input.incrementToken()) {
            String original = termAttr.toString();
            String hmacHex = HmacCalculator.hmacHex(hmacKey, original);
            termAttr.setEmpty().append(hmacHex);
            return true;
        }
        return false;
    }
}
