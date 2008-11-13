/**
 * Copyright 2007 The Apache Software Foundation
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hama.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.hbase.io.Cell;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Writable;
import org.apache.hama.SubMatrix;
import org.apache.hama.util.BytesUtil;
import org.apache.log4j.Logger;

public class BlockEntry implements Writable, Iterable<BlockEntry> {
  static final Logger LOG = Logger.getLogger(BlockEntry.class);
  protected byte[][] values;
  
  // We don't need this.
  @Deprecated
  protected long[] timestamps;

  /** For Writable compatibility */
  public BlockEntry() {
    values = null;
    timestamps = null;
  }

  public BlockEntry(Cell c) {
    this.values = new byte[1][];
    this.values[0] = c.getValue();
    this.timestamps = new long[1];
    this.timestamps[0] = c.getTimestamp();
  }

  public BlockEntry(SubMatrix value) throws IOException {
    this.values = new byte[1][];
    this.values[0] = BytesUtil.subMatrixToBytes(value);
    this.timestamps = new long[1];
    this.timestamps[0] = System.currentTimeMillis();
  }

  /**
   * @return the current VectorEntry's value
   * @throws IOException
   * @throws ClassNotFoundException
   */
  public SubMatrix getValue() throws IOException, ClassNotFoundException {
    return BytesUtil.bytesToSubMatrix(this.values[0]);
  }

  /** @return the current VectorEntry's timestamp */
  public long getTimestamp() {
    return timestamps[0];
  }

  /**
   * Create a new VectorEntry with a given value and timestamp. Used by HStore.
   * 
   * @param value
   * @param timestamp
   */
  public BlockEntry(byte[] value, long timestamp) {
    this.values = new byte[1][];
    this.values[0] = value;
    this.timestamps = new long[1];
    this.timestamps[0] = timestamp;
  }

  //
  // Writable
  //

  /** {@inheritDoc} */
  public void readFields(final DataInput in) throws IOException {
    int nvalues = in.readInt();
    this.timestamps = new long[nvalues];
    this.values = new byte[nvalues][];
    for (int i = 0; i < nvalues; i++) {
      this.timestamps[i] = in.readLong();
    }
    for (int i = 0; i < nvalues; i++) {
      this.values[i] = Bytes.readByteArray(in);
    }
  }

  /** {@inheritDoc} */
  public void write(final DataOutput out) throws IOException {
    out.writeInt(this.values.length);
    for (int i = 0; i < this.timestamps.length; i++) {
      out.writeLong(this.timestamps[i]);
    }
    for (int i = 0; i < this.values.length; i++) {
      Bytes.writeByteArray(out, this.values[i]);
    }
  }

  //
  // Iterable
  //

  /** {@inheritDoc} */
  public Iterator<BlockEntry> iterator() {
    return new BlockEntryIterator();
  }

  private class BlockEntryIterator implements Iterator<BlockEntry> {
    private int currentValue = -1;

    BlockEntryIterator() {
    }

    /** {@inheritDoc} */
    public boolean hasNext() {
      return currentValue < values.length;
    }

    /** {@inheritDoc} */
    public BlockEntry next() {
      currentValue += 1;
      return new BlockEntry(values[currentValue], timestamps[currentValue]);
    }

    /** {@inheritDoc} */
    public void remove() throws UnsupportedOperationException {
      throw new UnsupportedOperationException("remove is not supported");
    }
  }
}