// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/debug.proto

package org.tensorflow.framework;

public interface DebugTensorWatchOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.DebugTensorWatch)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Name of the node to watch.
   * </pre>
   *
   * <code>optional string node_name = 1;</code>
   */
  java.lang.String getNodeName();
  /**
   * <pre>
   * Name of the node to watch.
   * </pre>
   *
   * <code>optional string node_name = 1;</code>
   */
  com.google.protobuf.ByteString
      getNodeNameBytes();

  /**
   * <pre>
   * Output slot to watch.
   * The semantics of output_slot == -1 is that the node is only watched for
   * completion, but not for any output tensors. See NodeCompletionCallback
   * in debug_gateway.h.
   * TODO(cais): Implement this semantics.
   * </pre>
   *
   * <code>optional int32 output_slot = 2;</code>
   */
  int getOutputSlot();

  /**
   * <pre>
   * Name(s) of the debugging op(s).
   * One or more than one probes on a tensor.
   * e.g., {"DebugIdentity", "DebugNanCount"}
   * </pre>
   *
   * <code>repeated string debug_ops = 3;</code>
   */
  java.util.List<java.lang.String>
      getDebugOpsList();
  /**
   * <pre>
   * Name(s) of the debugging op(s).
   * One or more than one probes on a tensor.
   * e.g., {"DebugIdentity", "DebugNanCount"}
   * </pre>
   *
   * <code>repeated string debug_ops = 3;</code>
   */
  int getDebugOpsCount();
  /**
   * <pre>
   * Name(s) of the debugging op(s).
   * One or more than one probes on a tensor.
   * e.g., {"DebugIdentity", "DebugNanCount"}
   * </pre>
   *
   * <code>repeated string debug_ops = 3;</code>
   */
  java.lang.String getDebugOps(int index);
  /**
   * <pre>
   * Name(s) of the debugging op(s).
   * One or more than one probes on a tensor.
   * e.g., {"DebugIdentity", "DebugNanCount"}
   * </pre>
   *
   * <code>repeated string debug_ops = 3;</code>
   */
  com.google.protobuf.ByteString
      getDebugOpsBytes(int index);

  /**
   * <pre>
   * URL(s) for debug targets(s).
   *   E.g., "file:///foo/tfdbg_dump", "grpc://localhost:11011"
   * Each debug op listed in debug_ops will publish its output tensor (debug
   * signal) to all URLs in debug_urls.
   * </pre>
   *
   * <code>repeated string debug_urls = 4;</code>
   */
  java.util.List<java.lang.String>
      getDebugUrlsList();
  /**
   * <pre>
   * URL(s) for debug targets(s).
   *   E.g., "file:///foo/tfdbg_dump", "grpc://localhost:11011"
   * Each debug op listed in debug_ops will publish its output tensor (debug
   * signal) to all URLs in debug_urls.
   * </pre>
   *
   * <code>repeated string debug_urls = 4;</code>
   */
  int getDebugUrlsCount();
  /**
   * <pre>
   * URL(s) for debug targets(s).
   *   E.g., "file:///foo/tfdbg_dump", "grpc://localhost:11011"
   * Each debug op listed in debug_ops will publish its output tensor (debug
   * signal) to all URLs in debug_urls.
   * </pre>
   *
   * <code>repeated string debug_urls = 4;</code>
   */
  java.lang.String getDebugUrls(int index);
  /**
   * <pre>
   * URL(s) for debug targets(s).
   *   E.g., "file:///foo/tfdbg_dump", "grpc://localhost:11011"
   * Each debug op listed in debug_ops will publish its output tensor (debug
   * signal) to all URLs in debug_urls.
   * </pre>
   *
   * <code>repeated string debug_urls = 4;</code>
   */
  com.google.protobuf.ByteString
      getDebugUrlsBytes(int index);
}