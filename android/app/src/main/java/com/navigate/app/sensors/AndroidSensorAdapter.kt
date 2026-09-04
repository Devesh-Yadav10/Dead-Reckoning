package com.navigate.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.navigate.app.models.IMUSample

/**
 * AndroidSensorAdapter — Bridges Android SensorManager to NAVIGATE 2.0 10 Hz IMUSample stream.
 *
 * Responsibilities:
 * 1. Collects raw Accelerometer and Gyroscope sensor events.
 * 2. Regulates/decimates variable high-rate Android events (50-100 Hz) down to 10 Hz (100 ms).
 * 3. Transforms Android device sensor coordinates to Vehicle Forward-Left-Up (FLU) body frame.
 */
class AndroidSensorAdapter(
    context: Context,
    private val onSampleReady: (IMUSample) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private val accelSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Last known raw sensor readings
    private var lastAx = 0.0
    private var lastAy = 0.0
    private var lastAz = 9.81
    private var lastGx = 0.0
    private var lastGy = 0.0
    private var lastGz = 0.0

    private var lastEmittedTimestampSec = 0.0
    private val targetIntervalSec = 0.100 // 10 Hz target interval (100ms)

    var isRunning = false
        private set

    /**
     * Start sensor listeners.
     */
    fun start() {
        if (isRunning || sensorManager == null) return
        sensorManager.registerListener(this, accelSensor, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, gyroSensor, SensorManager.SENSOR_DELAY_GAME)
        lastEmittedTimestampSec = 0.0
        isRunning = true
    }

    /**
     * Stop sensor listeners.
     */
    fun stop() {
        if (!isRunning || sensorManager == null) return
        sensorManager.unregisterListener(this)
        isRunning = false
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return

        val timestampSec = event.timestamp / 1_000_000_000.0

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAx = event.values[0].toDouble()
                lastAy = event.values[1].toDouble()
                lastAz = event.values[2].toDouble()
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGx = event.values[0].toDouble()
                lastGy = event.values[1].toDouble()
                lastGz = event.values[2].toDouble()
            }
        }

        // 10 Hz Rate Regulation
        if (lastEmittedTimestampSec == 0.0 || (timestampSec - lastEmittedTimestampSec) >= targetIntervalSec) {
            val sample = createRawIMUSample(
                timestampSec = timestampSec,
                rawAx = lastAx,
                rawAy = lastAy,
                rawAz = lastAz,
                rawGx = lastGx,
                rawGy = lastGy,
                rawGz = lastGz
            )
            lastEmittedTimestampSec = timestampSec
            onSampleReady(sample)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // No-op for rate adapter
    }

    companion object {
        /**
         * Creates standard 6-axis raw phone frame IMUSample matching ML training data.
         *
         * Raw Phone Frame:
         *   - +X: Right
         *   - +Y: Top / Forward (Portrait)
         *   - +Z: Out of Screen
         */
        fun createRawIMUSample(
            timestampSec: Double,
            rawAx: Double,
            rawAy: Double,
            rawAz: Double,
            rawGx: Double,
            rawGy: Double,
            rawGz: Double
        ): IMUSample {
            return IMUSample(
                timestamp = timestampSec,
                ax = rawAx,
                ay = rawAy,
                az = rawAz,
                gx = rawGx,
                gy = rawGy,
                gz = rawGz
            )
        }

        /**
         * Transforms Android Phone coordinates to Vehicle Forward-Left-Up (FLU) frame.
         *
         * Standard Phone Portrait Mounting (Screen facing driver, top pointing up/forward):
         *   - Vehicle Forward (+X_v) = Phone Top (+Y_p)
         *   - Vehicle Left    (+Y_v) = Phone Left (-X_p)
         *   - Vehicle Up      (+Z_v) = Phone Out of Screen (+Z_p)
         */
        fun convertToVehicleBodyFrame(
            timestampSec: Double,
            rawAx: Double,
            rawAy: Double,
            rawAz: Double,
            rawGx: Double,
            rawGy: Double,
            rawGz: Double
        ): IMUSample {
            val fwdAcc = rawAy
            val leftAcc = -rawAx
            val upAcc = rawAz

            val fwdGyro = rawGy
            val leftGyro = -rawGx
            val upGyro = rawGz

            return IMUSample(
                timestamp = timestampSec,
                ax = fwdAcc,
                ay = leftAcc,
                az = upAcc,
                gx = fwdGyro,
                gy = leftGyro,
                gz = upGyro
            )
        }
    }
}

