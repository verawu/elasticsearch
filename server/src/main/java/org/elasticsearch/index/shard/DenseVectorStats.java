/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.shard;

import org.elasticsearch.TransportVersion;
import org.elasticsearch.TransportVersions;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.core.Nullable;
import org.elasticsearch.xcontent.ToXContentFragment;
import org.elasticsearch.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.Objects;

/**
 * Statistics about indexed dense vector
 */
public class DenseVectorStats implements Writeable, ToXContentFragment {

    private static final TransportVersion VECTOR_BUILD_STATS_VERSION = TransportVersions.VECTOR_BUILD_STATS_IN_DENSE_VECTOR;

    private long valueCount = 0;
    @Nullable
    private VectorBuildStats vectorBuildStats;

    public DenseVectorStats() {}

    public DenseVectorStats(long count) {
        this.valueCount = count;
    }

    public DenseVectorStats(StreamInput in) throws IOException {
        this.valueCount = in.readVLong();
        if (in.getTransportVersion().onOrAfter(VECTOR_BUILD_STATS_VERSION)) {
            this.vectorBuildStats = in.readOptionalWriteable(VectorBuildStats::new);
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeVLong(valueCount);
        if (out.getTransportVersion().onOrAfter(VECTOR_BUILD_STATS_VERSION)) {
            out.writeOptionalWriteable(vectorBuildStats);
        }
    }

    public void add(DenseVectorStats other) {
        if (other == null) {
            return;
        }
        this.valueCount += other.valueCount;
        if (other.vectorBuildStats != null) {
            this.vectorBuildStats = other.vectorBuildStats;
        }
    }

    public long getValueCount() {
        return valueCount;
    }

    @Nullable
    public VectorBuildStats getVectorBuildStats() {
        return vectorBuildStats;
    }

    public void setVectorBuildStats(VectorBuildStats vectorBuildStats) {
        this.vectorBuildStats = vectorBuildStats;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject(Fields.NAME);
        builder.field(Fields.VALUE_COUNT, valueCount);
        if (vectorBuildStats != null) {
            vectorBuildStats.toXContent(builder, params);
        }
        builder.endObject();
        return builder;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DenseVectorStats that = (DenseVectorStats) o;
        return valueCount == that.valueCount && Objects.equals(vectorBuildStats, that.vectorBuildStats);
    }

    @Override
    public int hashCode() {
        return Objects.hash(valueCount, vectorBuildStats);
    }

    static final class Fields {
        static final String NAME = "dense_vector";
        static final String VALUE_COUNT = "value_count";
    }

    public static class VectorBuildStats implements Writeable, ToXContentFragment {

        private final int pendingTasks;
        private final int runningTasks;
        private final long completedTasks;
        private final long totalBuildTimeMillis;
        private final long totalQueueTimeMillis;

        public VectorBuildStats(int pendingTasks, int runningTasks, long completedTasks, long totalBuildTimeMillis,
                                long totalQueueTimeMillis) {
            this.pendingTasks = pendingTasks;
            this.runningTasks = runningTasks;
            this.completedTasks = completedTasks;
            this.totalBuildTimeMillis = totalBuildTimeMillis;
            this.totalQueueTimeMillis = totalQueueTimeMillis;
        }

        public VectorBuildStats(StreamInput in) throws IOException {
            this.pendingTasks = in.readVInt();
            this.runningTasks = in.readVInt();
            this.completedTasks = in.readVLong();
            this.totalBuildTimeMillis = in.readVLong();
            this.totalQueueTimeMillis = in.readVLong();
        }

        @Override
        public void writeTo(StreamOutput out) throws IOException {
            out.writeVInt(pendingTasks);
            out.writeVInt(runningTasks);
            out.writeVLong(completedTasks);
            out.writeVLong(totalBuildTimeMillis);
            out.writeVLong(totalQueueTimeMillis);
        }

        @Override
        public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
            builder.startObject("vector_build");
            builder.field("pending_tasks", pendingTasks);
            builder.field("running_tasks", runningTasks);
            builder.field("completed_tasks", completedTasks);
            builder.field("total_build_time_millis", totalBuildTimeMillis);
            builder.field("total_queue_time_millis", totalQueueTimeMillis);
            builder.endObject();
            return builder;
        }

        public int getPendingTasks() {
            return pendingTasks;
        }

        public int getRunningTasks() {
            return runningTasks;
        }

        public long getCompletedTasks() {
            return completedTasks;
        }

        public long getTotalBuildTimeMillis() {
            return totalBuildTimeMillis;
        }

        public long getTotalQueueTimeMillis() {
            return totalQueueTimeMillis;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            VectorBuildStats that = (VectorBuildStats) o;
            return pendingTasks == that.pendingTasks
                && runningTasks == that.runningTasks
                && completedTasks == that.completedTasks
                && totalBuildTimeMillis == that.totalBuildTimeMillis
                && totalQueueTimeMillis == that.totalQueueTimeMillis;
        }

        @Override
        public int hashCode() {
            return Objects.hash(pendingTasks, runningTasks, completedTasks, totalBuildTimeMillis, totalQueueTimeMillis);
        }
    }
}
