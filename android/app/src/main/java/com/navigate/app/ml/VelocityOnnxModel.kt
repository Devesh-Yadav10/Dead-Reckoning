package com.navigate.app.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/**
 * VelocityOnnxModel — OnnxRuntime inference wrapper for VelocityModel V2.
 *
 * Runs 10 Hz window inference on 50-sample [ax, ay, az, gx, gy, gz] buffers.
 */
class VelocityOnnxModel(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
    private val config: VelocityModelConfig
) : AutoCloseable {

    private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())

    /**
     * Predicts scalar forward speed (in m/s) given a 50x6 IMU window.
     *
     * @param window 50 samples of [ax, ay, az, gx, gy, gz]
     * @return predicted vehicle forward speed in m/s (non-negative)
     */
    fun predictSpeed(window: List<FloatArray>): Float {
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
                val normSpeed = rawOutput[0][0]
                val speedMs = normSpeed * config.targetStd + config.targetMean
                return maxOf(0.0f, speedMs)
            }
        }
    }

    override fun close() {
        session.close()
    }
}

