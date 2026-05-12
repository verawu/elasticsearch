/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption;

import org.elasticsearch.common.settings.Setting;

import java.util.List;

public final class EncryptionSettings {

    public static final String KEY_PREFIX = "encryption.master_key.";

    public static final Setting<String> INDEX_ENCRYPTION_KEY_ID = Setting.simpleString(
        "index.encryption.key_id",
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    public static final Setting<List<String>> INDEX_ENCRYPTION_FIELDS = Setting.listSetting(
        "index.encryption.fields",
        List.of(),
        s -> s,
        Setting.Property.IndexScope,
        Setting.Property.Final
    );

    public static final String PIPELINE_NAME = "_field_encryption";

    public static final String DECRYPT_HEADER = "X-Decrypt-Authorized";

    private EncryptionSettings() {}

    public static List<Setting<?>> getSettings() {
        return List.of(INDEX_ENCRYPTION_KEY_ID, INDEX_ENCRYPTION_FIELDS);
    }
}
