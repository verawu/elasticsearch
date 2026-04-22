/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.ivfpq;

import org.elasticsearch.features.FeatureSpecification;
import org.elasticsearch.features.NodeFeature;

import java.util.Set;

public class IvfPqFeatures implements FeatureSpecification {

    public static final NodeFeature IVFPQ_VECTORS_FEATURE = new NodeFeature("ivfpq_vectors");

    @Override
    public Set<NodeFeature> getTestFeatures() {
        return Set.of(IVFPQ_VECTORS_FEATURE);
    }
}
