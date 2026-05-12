/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.plugin.encryption;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Static holder for KeyService, set during plugin component creation.
 * Allows token filter factories and processors to access key material.
 */
public final class KeyServiceHolder {

    private static final AtomicReference<KeyService> INSTANCE = new AtomicReference<>();

    private KeyServiceHolder() {}

    public static void set(KeyService keyService) {
        INSTANCE.set(keyService);
    }

    public static KeyService get() {
        KeyService service = INSTANCE.get();
        if (service == null) {
            throw new IllegalStateException("KeyService not initialized. Plugin may not be loaded.");
        }
        return service;
    }

    public static void clear() {
        KeyService service = INSTANCE.getAndSet(null);
        if (service != null) {
            service.close();
        }
    }
}
