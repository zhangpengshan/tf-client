package com.paypal.tf.client;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.NoSuchFileException;
import java.nio.file.Paths;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

/**
 * Tensorflow Java model client for dummy model
 */
public class TFModelJavaClient {

	public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException {
		ClassLoader classLoader = (TFModelJavaClient.class).getClassLoader();

		// A bug in win OS, need add winutils.exe and configure hadoop.home.dir
		File hadoopWinUtilFile = getAbsoluteFolder(classLoader, "hadoop-2.6.5");
		System.setProperty("hadoop.home.dir", hadoopWinUtilFile.toString());

		// get model file path, use parent path to load as SavedModelBundle
		File modelFile = getModelFile(classLoader, "tfmodel/spec2/saved_model.pb");
		try (SavedModelBundle bundle = SavedModelBundle.load(modelFile.getParent().toString(), "serve")) {
			float[][] inputs = new float[][] { new float[] { -0.515738f, 0.545345f, -1.020538f, 0.89946f, -0.397207f,
					-0.504616f, 0.435321f, -1.351367f, 0.70564f, -0.003322f, -2.806611f, -0.359407f, 0.923877f,
					-0.746241f, 0.0f, 0.71442f, -0.163535f, -0.826719f, 0.0f, -1.158023f, -0.049059f, 0.129129f,
					-0.118639f, 3.958756f, -0.8234f, -0.206282f, -0.38556f, 0.0f, -0.014673f, 0.027574f, -0.165786f,
					-0.057976f, 0.020935f, -0.653038f, -0.306683f, 2.383494f, 6.308608f, 7.876935f, 0.915056f,
					1.033469f, -1.271102f, -0.196439f, 2.321398f, 0.16255f, 0.117398f, 0.0f, 0.200878f, -0.009606f,
					5.329159f, 0.505601f, 0.107692f, -0.2032f, 0.101736f, 0.486856f, 1.397739f, 0.947251f, 0.0f,
					-0.098599f, 0.691554f, -0.078524f, 1.273427f, -0.037225f, 0.635263f, -0.709907f, -0.092292f,
					-0.095353f, 0.877157f, -0.651168f, -0.577033f, -0.243205f, -0.016549f, 0.0f, -0.012829f, -0.389046f,
					-0.375031f, 0.404269f, 0.0f, -0.338772f, 2.73259f, 1.028429f, 2.445034f, -0.15248f, -0.65f,
					-0.663064f, 0.0f, -1.007605f, 0.970459f, -0.06674f, -0.033648f, 8.070314f, -0.936844f, 0.0f,
					0.218354f, 0.0f, -0.259678f, 0.0f, -1.045223f, -0.767119f, -0.03372f, 0.444345f, -1.952245f,
					-0.276274f, -0.131328f, 0.193844f, -0.075098f, -0.036605f, -0.231917f, 2.198177f, 7.383884f,
					0.057013f, 0.0f, -0.076277f, 0.0f, -0.223626f, 0.0f, -0.201842f, 7.845792f, 0.106512f, -0.101737f,
					-0.50424f, -0.581858f, -0.005835f, 0.0f, 1.009325f, -0.16506f, 1.019598f, -0.391045f, -0.046094f,
					0.0f, 0.0f, 0.398988f, -0.352825f, 3.076625f, -0.515534f, 0.199322f, 0.0f, -0.279789f, 0.0f,
					-0.012211f, -0.044481f, -0.018219f, 0.166069f, 0.0f, -0.328976f, 0.552445f, -3.367463f, -0.004435f,
					0.395865f, 0.0f, 0.0f, -0.597038f, -0.152545f, -0.419586f, -0.039323f, 0.272912f, 0.0f, 0.0f,
					-0.153296f, 0.170611f, -4.340632f, -0.186318f, 0.864953f, 0.693297f, -0.032172f, 0.810094f,
					-0.360024f, -0.517794f, -0.413863f, 0.503393f, -0.444707f, -1.476417f, 1.05214f, 0.08001f,
					-1.722282f, -1.438142f, -0.239034f, -0.157187f, 5.188781f, -0.523153f, -0.213206f, -2.470411f,
					-0.628501f, 0.419582f, -1.398577f, -0.320261f, 2.223676f, 0.0f, -0.291793f, 4.485703f, 0.013283f,
					-0.56174f, 0.0f, 0.001182f, -0.133349f, -0.228256f, 0.629104f, 0.0f, 0.0f, 0.0f, 1.488518f,
					1.017088f, 1.081213f, -0.861844f, -0.343051f, 0.895915f, 0.443144f, -0.504448f, -0.229891f,
					-0.049247f, 0.0f, -0.330569f, 0.0f, -0.494514f, -2.042479f, -1.271284f, 0.0f, -1.200775f,
					-0.385822f, -0.04597f, -0.115198f, -0.144484f, 0.617743f, -0.652246f, 0.0f, 0.0f, 1.184211f, 0.0f,
					0.0f, 0.0f, 0.0f, 0.486267f, -1.950814f, -0.161482f, 0.243092f, -1.227096f, 1.294342f, 0.31503f,
					0.0f, -0.696342f, -0.23972f, 4.614801f, -0.313035f, 0.0f, 0.046658f, -0.322338f, 0.135278f,
					0.746147f, 0.0f, 0.09252f, -0.289131f, -0.06556f, -0.088676f, -0.32309f, 0.375648f, -1.196133f,
					1.637763f, -0.160962f, -0.190001f, 0.250074f, 0.311723f, -0.028751f, 0.0f, -0.754593f, 1.675634f,
					-0.236019f, -1.867039f, 0.563342f, 0.858923f, 0.0f, -0.429037f, 0.0f, -2.660938f, -0.050719f,
					-0.167224f, -0.098971f, -0.205477f, 0.455025f, 0.0f, 0.178238f, 0.245741f, -1.768768f, -0.094145f,
					-0.570002f, -0.049828f, -0.238787f, -0.12209f, -0.236684f, -0.500676f, 0.199192f, 0.190035f, 0.0f,
					0.007894f, -0.403599f, -2.241858f } };

			// create input tensor which is 2-dimensions
			Tensor input = Tensor.create(inputs);

			// run inference by feeding input, be careful of input and output op name
			Tensor result = bundle.session().runner().feed("dense_1_input", input).fetch("dense_2/Sigmoid").run()
					.get(0);

			System.out.println(result.copyTo(new float[1][1])[0][0]);
		}
	}

	private static File getModelFile(ClassLoader classLoader, String modelFile)
			throws NoSuchFileException, URISyntaxException {
		String protoPath = modelFile;
		URL protoResource = classLoader.getResource(protoPath);
		if (protoResource == null) {
			throw new NoSuchFileException(protoPath);
		}

		File protoFile = (Paths.get(protoResource.toURI())).toFile();
		return protoFile;
	}

	private static File getAbsoluteFolder(ClassLoader classLoader, String currFolder)
			throws NoSuchFileException, URISyntaxException {
		String protoPath = currFolder;
		URL protoResource = classLoader.getResource(protoPath);
		if (protoResource == null) {
			throw new NoSuchFileException(protoPath);
		}
		File protoFile = (Paths.get(protoResource.toURI())).toFile();
		return protoFile;
	}

}
