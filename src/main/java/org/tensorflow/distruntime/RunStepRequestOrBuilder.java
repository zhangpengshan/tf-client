// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/master.proto

package org.tensorflow.distruntime;

public interface RunStepRequestOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.RunStepRequest)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * REQUIRED: session_handle must be returned by a CreateSession call
   * to the same master service.
   * </pre>
   *
   * <code>optional string session_handle = 1;</code>
   */
  java.lang.String getSessionHandle();
  /**
   * <pre>
   * REQUIRED: session_handle must be returned by a CreateSession call
   * to the same master service.
   * </pre>
   *
   * <code>optional string session_handle = 1;</code>
   */
  com.google.protobuf.ByteString
      getSessionHandleBytes();

  /**
   * <pre>
   * Tensors to be fed in the step. Each feed is a named tensor.
   * </pre>
   *
   * <code>repeated .tensorflow.NamedTensorProto feed = 2;</code>
   */
  java.util.List<org.tensorflow.framework.NamedTensorProto> 
      getFeedList();
  /**
   * <pre>
   * Tensors to be fed in the step. Each feed is a named tensor.
   * </pre>
   *
   * <code>repeated .tensorflow.NamedTensorProto feed = 2;</code>
   */
  org.tensorflow.framework.NamedTensorProto getFeed(int index);
  /**
   * <pre>
   * Tensors to be fed in the step. Each feed is a named tensor.
   * </pre>
   *
   * <code>repeated .tensorflow.NamedTensorProto feed = 2;</code>
   */
  int getFeedCount();
  /**
   * <pre>
   * Tensors to be fed in the step. Each feed is a named tensor.
   * </pre>
   *
   * <code>repeated .tensorflow.NamedTensorProto feed = 2;</code>
   */
  java.util.List<? extends org.tensorflow.framework.NamedTensorProtoOrBuilder> 
      getFeedOrBuilderList();
  /**
   * <pre>
   * Tensors to be fed in the step. Each feed is a named tensor.
   * </pre>
   *
   * <code>repeated .tensorflow.NamedTensorProto feed = 2;</code>
   */
  org.tensorflow.framework.NamedTensorProtoOrBuilder getFeedOrBuilder(
      int index);

  /**
   * <pre>
   * Fetches. A list of tensor names. The caller expects a tensor to
   * be returned for each fetch[i] (see RunStepResponse.tensor). The
   * order of specified fetches does not change the execution order.
   * </pre>
   *
   * <code>repeated string fetch = 3;</code>
   */
  java.util.List<java.lang.String>
      getFetchList();
  /**
   * <pre>
   * Fetches. A list of tensor names. The caller expects a tensor to
   * be returned for each fetch[i] (see RunStepResponse.tensor). The
   * order of specified fetches does not change the execution order.
   * </pre>
   *
   * <code>repeated string fetch = 3;</code>
   */
  int getFetchCount();
  /**
   * <pre>
   * Fetches. A list of tensor names. The caller expects a tensor to
   * be returned for each fetch[i] (see RunStepResponse.tensor). The
   * order of specified fetches does not change the execution order.
   * </pre>
   *
   * <code>repeated string fetch = 3;</code>
   */
  java.lang.String getFetch(int index);
  /**
   * <pre>
   * Fetches. A list of tensor names. The caller expects a tensor to
   * be returned for each fetch[i] (see RunStepResponse.tensor). The
   * order of specified fetches does not change the execution order.
   * </pre>
   *
   * <code>repeated string fetch = 3;</code>
   */
  com.google.protobuf.ByteString
      getFetchBytes(int index);

  /**
   * <pre>
   * Target Nodes. A list of node names. The named nodes will be run
   * to but their outputs will not be fetched.
   * </pre>
   *
   * <code>repeated string target = 4;</code>
   */
  java.util.List<java.lang.String>
      getTargetList();
  /**
   * <pre>
   * Target Nodes. A list of node names. The named nodes will be run
   * to but their outputs will not be fetched.
   * </pre>
   *
   * <code>repeated string target = 4;</code>
   */
  int getTargetCount();
  /**
   * <pre>
   * Target Nodes. A list of node names. The named nodes will be run
   * to but their outputs will not be fetched.
   * </pre>
   *
   * <code>repeated string target = 4;</code>
   */
  java.lang.String getTarget(int index);
  /**
   * <pre>
   * Target Nodes. A list of node names. The named nodes will be run
   * to but their outputs will not be fetched.
   * </pre>
   *
   * <code>repeated string target = 4;</code>
   */
  com.google.protobuf.ByteString
      getTargetBytes(int index);

  /**
   * <pre>
   * Options for the run call.
   * </pre>
   *
   * <code>optional .tensorflow.RunOptions options = 5;</code>
   */
  boolean hasOptions();
  /**
   * <pre>
   * Options for the run call.
   * </pre>
   *
   * <code>optional .tensorflow.RunOptions options = 5;</code>
   */
  org.tensorflow.framework.RunOptions getOptions();
  /**
   * <pre>
   * Options for the run call.
   * </pre>
   *
   * <code>optional .tensorflow.RunOptions options = 5;</code>
   */
  org.tensorflow.framework.RunOptionsOrBuilder getOptionsOrBuilder();

  /**
   * <pre>
   * Partial run handle (optional). If specified, this will be a partial run
   * execution, run up to the specified fetches.
   * </pre>
   *
   * <code>optional string partial_run_handle = 6;</code>
   */
  java.lang.String getPartialRunHandle();
  /**
   * <pre>
   * Partial run handle (optional). If specified, this will be a partial run
   * execution, run up to the specified fetches.
   * </pre>
   *
   * <code>optional string partial_run_handle = 6;</code>
   */
  com.google.protobuf.ByteString
      getPartialRunHandleBytes();
}
