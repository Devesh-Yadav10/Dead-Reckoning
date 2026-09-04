package com.navigate.app.ml

import org.json.JSONObject

data class VelocityModelConfig(
    val file: String,
    val inputName: String,
    val outputName: String,
    val imuMean: FloatArray,
    val imuStd: FloatArray,
    val targetMean: Float,
    val targetStd: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as VelocityModelConfig
        return file == other.file &&
                inputName == other.inputName &&
                outputName == other.outputName &&
                imuMean.contentEquals(other.imuMean) &&
                imuStd.contentEquals(other.imuStd) &&
                targetMean == other.targetMean &&
                targetStd == other.targetStd
    }

    override fun hashCode(): Int {
        var result = file.hashCode()
        result = 31 * result + inputName.hashCode()
        result = 31 * result + outputName.hashCode()
        result = 31 * result + imuMean.contentHashCode()
        result = 31 * result + imuStd.contentHashCode()
        result = 31 * result + targetMean.hashCode()
        result = 31 * result + targetStd.hashCode()
        return result
    }
}

data class AttitudeModelConfig(
    val file: String,
    val inputName: String,
    val outputName: String,
    val imuMean: FloatArray,
    val imuStd: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as AttitudeModelConfig
        return file == other.file &&
                inputName == other.inputName &&
                outputName == other.outputName &&
                imuMean.contentEquals(other.imuMean) &&
                imuStd.contentEquals(other.imuStd)
    }

    override fun hashCode(): Int {
        var result = file.hashCode()
        result = 31 * result + inputName.hashCode()
        result = 31 * result + outputName.hashCode()
        result = 31 * result + imuMean.contentHashCode()
        result = 31 * result + imuStd.contentHashCode()
        return result
    }
}

data class ModelMetadata(
    val velocityConfig: VelocityModelConfig,
    val attitudeConfig: AttitudeModelConfig
) {
    companion object {
        fun fromJson(jsonStr: String): ModelMetadata {
            val root = JSONObject(jsonStr)
            val velObj = root.getJSONObject("velocity_model")
            val attObj = root.getJSONObject("attitude_model")

            val velMean = parseJsonFloatArray(velObj.getJSONArray("imu_mean"))
            val velStd = parseJsonFloatArray(velObj.getJSONArray("imu_std"))
            val attMean = parseJsonFloatArray(attObj.getJSONArray("imu_mean"))
            val attStd = parseJsonFloatArray(attObj.getJSONArray("imu_std"))

            val velocityConfig = VelocityModelConfig(
                file = velObj.optString("file", "velocity_model_v2.onnx"),
                inputName = velObj.optString("input_name", "imu_input"),
                outputName = velObj.optString("output_name", "velocity_output"),
                imuMean = velMean,
                imuStd = velStd,
                targetMean = velObj.optDouble("target_mean", 11.5).toFloat(),
                targetStd = velObj.optDouble("target_std", 8.6).toFloat()
            )

            val attitudeConfig = AttitudeModelConfig(
                file = attObj.optString("file", "attitude_model.onnx"),
                inputName = attObj.optString("input_name", "imu_input"),
                outputName = attObj.optString("output_name", "attitude_output"),
                imuMean = attMean,
                imuStd = attStd
            )

            return ModelMetadata(velocityConfig, attitudeConfig)
        }

        private fun parseJsonFloatArray(jsonArray: org.json.JSONArray): FloatArray {
            val array = FloatArray(jsonArray.length())
            for (i in 0 until jsonArray.length()) {
                array[i] = jsonArray.getDouble(i).toFloat()
            }
            return array
        }
    }
}

