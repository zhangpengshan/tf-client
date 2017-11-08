// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/config.proto

package org.tensorflow.framework;

public interface ThreadPoolOptionProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.ThreadPoolOptionProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * The number of threads in the pool.
   * 0 means the system picks a value based on where this option proto is used
   * (see the declaration of the specific field for more info).
   * </pre>
   *
   * <code>optional int32 num_threads = 1;</code>
   */
  int getNumThreads();
}
