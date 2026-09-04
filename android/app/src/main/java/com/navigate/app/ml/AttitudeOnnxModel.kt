package com.navigate.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * AttitudeOnnxModel — OnnxRuntime inference wrapper for AttitudeModel.
 *
 * Runs 10 Hz window inference on 50-sample [ax, ay, az, gx, gy, gz] buffers.
 * Returns unit quaternion [qw, qx, qy, qz] with qw >= 0.
 */
class AttitudeOnnxModel(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
    private val config: AttitudeModelConfig
) : AutoCloseable {

    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())

    /**
     * Predicts unit quaternion [qw, qx, qy, qz] given a 50x6 IMU window.
     *
     * @param window 50 samples of [ax, ay, az, gx, gy, gz]
     * @return 4-element FloatArray [qw, qx, qy, qz] with norm=1 and qw >= 0
     */
    fun predictQuaternion(window: List<FloatArray>): FloatArray {
        require(window.size == 50) { "Window size must be exactly 50 samples" }

        val flatNormalized = FloatArray(50 * 6)
        var idx = 0
        for (sample in window) {
            require(sample.size == 6) { "Sample size must be 6 channels" }
            for (c in 0 until 6) {
                val mean = config.imuMean[c]
                val std = if (config.imuStd[c] > 1e-6f) config.imuStd[c] else 1.0f
                flatNormalized[idx++] = (sample[c] - mean) / std
            }
        }

        val shape = longArrayOf(1, 50, 6)
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatNormalized), shape)

        tensor.use { inTensor ->
            val inputs = mapOf(config.inputName to inTensor)
            session.run(inputs).use { results ->
                val outTensor = results.get(0) as OnnxTensor
                @Suppress("UNCHECKED_CAST")
                val rawOutput = outTensor.value as Array<FloatArray>
                val q = rawOutput[0].copyOf()

                // Enforce unit normalization
                val norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
                if (norm > 1e-6f) {
                    q[0] /= norm
                    q[1] /= norm
                    q[2] /= norm
                    q[3] /= norm
                } else {
                    q[0] = 1.0f
                    q[1] = 0.0f
                    q[2] = 0.0f
                    q[3] = 0.0f
                }

                // Enforce non-negative qw canonical convention
                if (q[0] < 0.0f) {
                    q[0] = -q[0]
                    q[1] = -q[1]
                    q[2] = -q[2]
                    q[3] = -q[3]
                }

                return q
            }
        }
    }

    override fun close() {
        session.close()
    }
}

