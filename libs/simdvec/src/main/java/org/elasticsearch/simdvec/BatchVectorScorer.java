/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec;

import java.io.IOException;

/**
 * A scorer that can compute distances for a contiguous range of ordinals in a single native batch call.
 */
public interface BatchVectorScorer {

    int MAX_BATCH_SIZE = 64;

    /**
     * Scores a contiguous range of ordinals [startOrd, startOrd+count) in a single native batch call.
     *
     * @param startOrd the first ordinal to score
     * @param count    the number of ordinals to score (must be &lt;= {@link #MAX_BATCH_SIZE})
     * @param results  output array of at least {@code count} floats
     * @return the number of ordinals scored, or -1 if the range cannot be batch-scored
     */
    int scoreBatch(int startOrd, int count, float[] results) throws IOException;
}
