package com.paypal.tf.client;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

/**
 * Tf serving client for spec2 model
 */
public class Spec2Client {

	private final ManagedChannel channel;

	private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

	// Initialize gRPC client
	public Spec2Client(String host, int port) {
		channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
		blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		// host and port setting
		String host = "shifu.ml";
		int port = 8080;

		if (args.length == 1) {
			String[] server_pair = args[0].split("=");
			if (!server_pair[0].equals("--server")) {
				System.out.println("you can only specify server address, no other args");
				return;
			}
			String[] server = server_pair[1].split(":");
			host = server[0];
			port = Integer.parseInt(server[1]);
		}

		// model name and model version setting, this should be consistent with
		// TF.serving name and version
		String modelName = "spec2";
		long modelVersion = 1;

		// Run predict client to send request
		Spec2Client client = new Spec2Client(host, port);

		try {
			client.doPredict(modelName, modelVersion);
		} finally {
			client.shutdown();
		}
	}

	public void shutdown() throws InterruptedException {
		channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
	}

	public void doPredict(String modelName, long modelVersion) throws IOException {
		// float [] inputs = new float[] {
		// -0.515738f,0.545345f,-1.020538f,0.89946f,-0.397207f,-0.504616f,0.435321f,0.596208f,-0.032661f,-0.244839f,-0.147976f,-0.359407f,-0.682663f,-0.746241f,0.784509f,0.0f,-0.137027f,1.080023f,0.0f,-1.158023f,-0.049059f,0.96767f,-0.118769f,-0.365039f,-0.8234f,-0.961708f,-0.38556f,0.0f,-0.015494f,-0.042615f,-0.481345f,-0.057976f,-0.070485f,5.998367f,-0.306683f,-2.049667f,-0.686346f,-0.177641f,0.915056f,1.033469f,0.371473f,-0.196439f,2.770148f,0.091247f,0.117398f,0.0f,-0.032085f,-0.056804f,0.69983f,0.505601f,0.129331f,-0.503914f,-1.004657f,-2.003711f,1.397739f,0.947251f,0.0f,-0.619213f,-0.424729f,-0.078524f,1.041013f,0.0f,-0.633368f,-0.76241f,-0.092292f,-0.095353f,-1.083408f,-0.123873f,2.949627f,0.224776f,-0.137918f,1.191668f,-0.010542f,6.425045f,-0.375031f,-0.512486f,-0.775635f,-0.053548f,2.73259f,1.028429f,-0.773128f,-0.39908f,2.405556f,-0.663064f,0.0f,-1.007605f,-0.949112f,-0.141172f,0.0f,1.426381f,-0.936844f,0.051689f,-1.832805f,0.0f,-0.259705f,0.0f,-1.045223f,1.303579f,0.51585f,-0.310148f,0.529874f,-0.293349f,-0.131328f,-0.117912f,-0.125103f,-0.036605f,-0.231917f,1.675017f,-0.142297f,-0.199939f,0.0f,-0.076332f,0.0f,-0.223626f,0.0f,-0.034986f,-0.027436f,-0.200328f,-0.101737f,-0.50424f,0.773133f,-0.005835f,0.0f,1.009325f,-0.16506f,1.019598f,-0.391045f,0.008311f,1.961781f,0.0f,-1.913328f,2.178621f,0.341719f,-0.515534f,3.201499f,0.0f,-0.279789f,-0.386626f,1.190797f,-0.020746f,1.358793f,0.0f,-0.74981f,-0.328976f,0.0f,0.299262f,-0.039445f,-0.682146f,-0.088832f,0.0f,-0.963543f,-0.152545f,0.918142f,-0.039323f,0.272912f,0.0f,-0.038775f,-0.007346f,0.170611f,0.253749f,1.042686f,-0.419488f,-0.344124f,-0.032172f,-0.118109f,-0.360024f,-0.517792f,-1.099493f,-0.686643f,-0.444707f,0.502742f,1.05214f,0.08001f,0.58064f,-0.946527f,-0.239034f,-0.157187f,-0.39263f,-0.523153f,0.88642f,0.042475f,1.561427f,-0.392475f,0.562387f,-0.320261f,0.062f,0.0f,2.850361f,0.0f,0.318489f,-0.566214f,0.0f,0.001182f,-0.15326f,-0.228256f,-0.502305f,-0.07016f,-0.326316f,0.0f,-0.665346f,1.01709f,-0.369425f,-0.91232f,-0.343051f,-1.118201f,-0.367585f,0.0f,0.055209f,-0.052384f,0.0f,-0.330569f,0.0f,1.174175f,0.489618f,-1.271284f,1.395031f,-1.200775f,-0.618529f,0.015984f,-0.115198f,-0.144484f,3.136933f,-0.619667f,-0.082307f,-0.578376f,-0.364657f,0.0f,0.0f,-0.705918f,0.0f,0.486267f,0.0f,-0.161482f,0.174113f,1.030844f,-0.06132f,0.31503f,-0.225715f,0.529666f,-0.23972f,0.0f,-0.313061f,-0.144029f,-0.062623f,-0.322338f,0.031821f,-0.258972f,0.0f,-0.5599f,-0.289131f,-0.06556f,-0.088676f,-0.32309f,0.0f,-0.801629f,-0.61059f,-0.160962f,0.0f,0.250074f,0.276144f,-0.305019f,0.0f,-0.754593f,-0.04655f,0.0f,-0.126815f,0.563342f,0.858891f,-0.700156f,-0.423705f,0.0f,0.375806f,-0.076465f,-0.167271f,-0.098971f,-0.205477f,0.0f,0.315402f,-0.042309f,-0.532462f,0.582134f,-0.483083f,-0.570002f,-0.049828f,-0.238787f,-0.12209f,-0.025044f,2.31813f,0.0f,-0.031639f,0.0f,0.330572f,2.151686f,0.0f
		// };

		float[] inputs = new float[] { -0.515738f, 0.545345f, -1.020538f, 0.89946f, -0.397207f, -0.504616f, 0.435321f,
				-1.351367f, 0.70564f, -0.003322f, -2.806611f, -0.359407f, 0.923877f, -0.746241f, 0.0f, 0.71442f,
				-0.163535f, -0.826719f, 0.0f, -1.158023f, -0.049059f, 0.129129f, -0.118639f, 3.958756f, -0.8234f,
				-0.206282f, -0.38556f, 0.0f, -0.014673f, 0.027574f, -0.165786f, -0.057976f, 0.020935f, -0.653038f,
				-0.306683f, 2.383494f, 6.308608f, 7.876935f, 0.915056f, 1.033469f, -1.271102f, -0.196439f, 2.321398f,
				0.16255f, 0.117398f, 0.0f, 0.200878f, -0.009606f, 5.329159f, 0.505601f, 0.107692f, -0.2032f, 0.101736f,
				0.486856f, 1.397739f, 0.947251f, 0.0f, -0.098599f, 0.691554f, -0.078524f, 1.273427f, -0.037225f,
				0.635263f, -0.709907f, -0.092292f, -0.095353f, 0.877157f, -0.651168f, -0.577033f, -0.243205f,
				-0.016549f, 0.0f, -0.012829f, -0.389046f, -0.375031f, 0.404269f, 0.0f, -0.338772f, 2.73259f, 1.028429f,
				2.445034f, -0.15248f, -0.65f, -0.663064f, 0.0f, -1.007605f, 0.970459f, -0.06674f, -0.033648f, 8.070314f,
				-0.936844f, 0.0f, 0.218354f, 0.0f, -0.259678f, 0.0f, -1.045223f, -0.767119f, -0.03372f, 0.444345f,
				-1.952245f, -0.276274f, -0.131328f, 0.193844f, -0.075098f, -0.036605f, -0.231917f, 2.198177f, 7.383884f,
				0.057013f, 0.0f, -0.076277f, 0.0f, -0.223626f, 0.0f, -0.201842f, 7.845792f, 0.106512f, -0.101737f,
				-0.50424f, -0.581858f, -0.005835f, 0.0f, 1.009325f, -0.16506f, 1.019598f, -0.391045f, -0.046094f, 0.0f,
				0.0f, 0.398988f, -0.352825f, 3.076625f, -0.515534f, 0.199322f, 0.0f, -0.279789f, 0.0f, -0.012211f,
				-0.044481f, -0.018219f, 0.166069f, 0.0f, -0.328976f, 0.552445f, -3.367463f, -0.004435f, 0.395865f, 0.0f,
				0.0f, -0.597038f, -0.152545f, -0.419586f, -0.039323f, 0.272912f, 0.0f, 0.0f, -0.153296f, 0.170611f,
				-4.340632f, -0.186318f, 0.864953f, 0.693297f, -0.032172f, 0.810094f, -0.360024f, -0.517794f, -0.413863f,
				0.503393f, -0.444707f, -1.476417f, 1.05214f, 0.08001f, -1.722282f, -1.438142f, -0.239034f, -0.157187f,
				5.188781f, -0.523153f, -0.213206f, -2.470411f, -0.628501f, 0.419582f, -1.398577f, -0.320261f, 2.223676f,
				0.0f, -0.291793f, 4.485703f, 0.013283f, -0.56174f, 0.0f, 0.001182f, -0.133349f, -0.228256f, 0.629104f,
				0.0f, 0.0f, 0.0f, 1.488518f, 1.017088f, 1.081213f, -0.861844f, -0.343051f, 0.895915f, 0.443144f,
				-0.504448f, -0.229891f, -0.049247f, 0.0f, -0.330569f, 0.0f, -0.494514f, -2.042479f, -1.271284f, 0.0f,
				-1.200775f, -0.385822f, -0.04597f, -0.115198f, -0.144484f, 0.617743f, -0.652246f, 0.0f, 0.0f, 1.184211f,
				0.0f, 0.0f, 0.0f, 0.0f, 0.486267f, -1.950814f, -0.161482f, 0.243092f, -1.227096f, 1.294342f, 0.31503f,
				0.0f, -0.696342f, -0.23972f, 4.614801f, -0.313035f, 0.0f, 0.046658f, -0.322338f, 0.135278f, 0.746147f,
				0.0f, 0.09252f, -0.289131f, -0.06556f, -0.088676f, -0.32309f, 0.375648f, -1.196133f, 1.637763f,
				-0.160962f, -0.190001f, 0.250074f, 0.311723f, -0.028751f, 0.0f, -0.754593f, 1.675634f, -0.236019f,
				-1.867039f, 0.563342f, 0.858923f, 0.0f, -0.429037f, 0.0f, -2.660938f, -0.050719f, -0.167224f,
				-0.098971f, -0.205477f, 0.455025f, 0.0f, 0.178238f, 0.245741f, -1.768768f, -0.094145f, -0.570002f,
				-0.049828f, -0.238787f, -0.12209f, -0.236684f, -0.500676f, 0.199192f, 0.190035f, 0.0f, 0.007894f,
				-0.403599f, -2.241858f };

		// input tensor building
		TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();
		for (int j = 0; j < inputs.length; ++j) {
			featuresTensorBuilder.addFloatVal(inputs[j]);
		}
		TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
		TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder().setSize(inputs.length).build();
		TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).addDim(featuresDim2)
				.build();
		featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_FLOAT).setTensorShape(featuresShape);
		TensorProto featuresTensorProto = featuresTensorBuilder.build();

		// Generate gRPC request, signature inputs name should be correct or exceptions
		com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion)
				.build();
		Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version)
				.setSignatureName("predict").build();
		Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec)
				.putInputs("inputs", featuresTensorProto).build();

		// Request gRPC server
		try {
			long start = System.currentTimeMillis();
			Predict.PredictResponse response = blockingStub.predict(request);
			java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputsMap();
			System.out.println(outputs.toString());
			System.out.println((System.currentTimeMillis() - start) + "ms");
		} catch (StatusRuntimeException e) {
			e.printStackTrace();
		}
	}
}
