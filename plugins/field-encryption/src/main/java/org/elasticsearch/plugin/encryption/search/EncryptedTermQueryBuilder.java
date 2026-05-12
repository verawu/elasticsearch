/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption.search;

import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.index.Term;
import org.elasticsearch.TransportVersion;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.index.query.AbstractQueryBuilder;
import org.elasticsearch.index.query.QueryRewriteContext;
import org.elasticsearch.index.query.SearchExecutionContext;
import org.elasticsearch.plugin.encryption.KeyServiceHolder;
import org.elasticsearch.plugin.encryption.crypto.HmacCalculator;
import org.elasticsearch.xcontent.ParseField;
import org.elasticsearch.xcontent.XContentBuilder;
import org.elasticsearch.xcontent.XContentParser;

import java.io.IOException;
import java.util.Objects;

/**
 * A query builder that HMACs the query value before performing a term lookup,
 * enabling search over HMAC-encrypted inverted index terms.
 *
 * Usage:
 * {
 *   "encrypted_term": {
 *     "field": "email",
 *     "value": "alice@example.com",
 *     "key_id": "pii-key"
 *   }
 * }
 */
public class EncryptedTermQueryBuilder extends AbstractQueryBuilder<EncryptedTermQueryBuilder> {

    public static final String NAME = "encrypted_term";
    public static final ParseField FIELD_FIELD = new ParseField("field");
    public static final ParseField VALUE_FIELD = new ParseField("value");
    public static final ParseField KEY_ID_FIELD = new ParseField("key_id");

    private final String field;
    private final String value;
    private final String keyId;

    public EncryptedTermQueryBuilder(String field, String value, String keyId) {
        this.field = Objects.requireNonNull(field);
        this.value = Objects.requireNonNull(value);
        this.keyId = Objects.requireNonNull(keyId);
    }

    public EncryptedTermQueryBuilder(StreamInput in) throws IOException {
        super(in);
        this.field = in.readString();
        this.value = in.readString();
        this.keyId = in.readString();
    }

    @Override
    protected void doWriteTo(StreamOutput out) throws IOException {
        out.writeString(field);
        out.writeString(value);
        out.writeString(keyId);
    }

    @Override
    protected void doXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(NAME);
        builder.field(FIELD_FIELD.getPreferredName(), field);
        builder.field(VALUE_FIELD.getPreferredName(), value);
        builder.field(KEY_ID_FIELD.getPreferredName(), keyId);
        builder.endObject();
    }

    @Override
    protected Query doToQuery(SearchExecutionContext context) throws IOException {
        byte[] hmacKey = KeyServiceHolder.get().getHmacKey(keyId);
        String hmacValue = HmacCalculator.hmacHex(hmacKey, value);
        return new TermQuery(new Term(field, hmacValue));
    }

    @Override
    protected boolean doEquals(EncryptedTermQueryBuilder other) {
        return Objects.equals(field, other.field) && Objects.equals(value, other.value) && Objects.equals(keyId, other.keyId);
    }

    @Override
    protected int doHashCode() {
        return Objects.hash(field, value, keyId);
    }

    @Override
    public String getWriteableName() {
        return NAME;
    }

    @Override
    public TransportVersion getMinimalSupportedVersion() {
        return TransportVersion.zero();
    }

    public static EncryptedTermQueryBuilder fromXContent(XContentParser parser) throws IOException {
        String field = null;
        String value = null;
        String keyId = null;

        XContentParser.Token token = parser.nextToken();
        if (token != XContentParser.Token.START_OBJECT) {
            throw new IllegalArgumentException("expected start object");
        }

        while ((token = parser.nextToken()) != XContentParser.Token.END_OBJECT) {
            if (token == XContentParser.Token.FIELD_NAME) {
                String fieldName = parser.currentName();
                parser.nextToken();
                if (FIELD_FIELD.match(fieldName, parser.getDeprecationHandler())) {
                    field = parser.text();
                } else if (VALUE_FIELD.match(fieldName, parser.getDeprecationHandler())) {
                    value = parser.text();
                } else if (KEY_ID_FIELD.match(fieldName, parser.getDeprecationHandler())) {
                    keyId = parser.text();
                }
            }
        }

        if (field == null || value == null || keyId == null) {
            throw new IllegalArgumentException("encrypted_term query requires [field], [value], and [key_id]");
        }
        return new EncryptedTermQueryBuilder(field, value, keyId);
    }
}
