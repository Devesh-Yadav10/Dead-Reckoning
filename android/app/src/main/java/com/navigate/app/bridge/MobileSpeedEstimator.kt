package com.navigate.app.bridge

import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * MobileSpeedEstimator — Robust multi-source speed estimator for mobile devices.
 *
 * Combines:
 *   1. Zero Velocity Detection (ZVD) via 3D specific force and angular velocity magnitude.
 *   2. GNSS speed validation, low-speed deadband, and spike/rate-of-change rejection.
 *   3. AI VelocityModel speed fusion during normal GNSS and outage conditions.
 *   4. Safe cold-start handling preventing accelerometer drift at rest.
 */
class MobileSpeedEstimator(
    private val accelStationaryThreshold: Double = 0.50, // m/s^2 deviation from 1g (9.80665 m/s^2)
    private val gyroStationaryThreshold: Double = 0.12,  // rad/s (~6.88 deg/s)
    private val stationaryEnterCount: Int = 5,           // 0.5s at 10 Hz
    private val stationaryExitCount: Int = 2,            // 0.2s at 10 Hz
    private val maxPlausibleAccel: Double = 6.0,         // m/s^2 (~0.61g max vehicle rate of speed change)
    private val gnssDeadbandMs: Double = 0.4             // m/s (~1.44 km/h) low-speed noise deadband
) {
    private val standardGravity = 9.80665

    // Stationary Detector State
    var isStationary: Boolean = true
        private set
    private var consecutiveStationarySamples = 0
    private var consecutiveMotionSamples = 0

    // Speed Estimates
    var rawGnssSpeed: Double? = null
        private set
    var filteredGnssSpeed: Double = 0.0
        private set
    var aiPredictedSpeed: Double = 0.0
        private set
    var currentSpeedMs: Double = 0.0
        private set

    private var lastGnssTimestamp: Double? = null

    /**
     * Resets all internal filter and detector states.
     */
    fun reset() {
        isStationary = true
        consecutiveStationarySamples = 0
        consecutiveMotionSamples = 0
        rawGnssSpeed = null
        filteredGnssSpeed = 0.0
        aiPredictedSpeed = 0.0
        currentSpeedMs = 0.0
        lastGnssTimestamp = null
    }

    /**
     * Updates stationary detector with streaming 10 Hz IMU sample and optional linear acceleration magnitude.
     */
    fun updateIMU(sample: IMUSample, linearAccelMag: Double = 0.0) {
        val accelMag = sqrt(sample.ax * sample.ax + sample.ay * sample.ay + sample.az * sample.az)
        val gyroMag = sqrt(sample.gx * sample.gx + sample.gy * sample.gy + sample.gz * sample.gz)

        val accelDiff = abs(accelMag - standardGravity)
        val sampleIsStationary = (accelDiff < accelStationaryThreshold && linearAccelMag < 0.35) && (gyroMag < gyroStationaryThreshold)

        if (sampleIsStationary) {
            consecutiveStationarySamples++
            consecutiveMotionSamples = 0
            if (consecutiveStationarySamples >= stationaryEnterCount) {
                isStationary = true
            }
        } else {
            consecutiveMotionSamples++
            consecutiveStationarySamples = 0
            if (consecutiveMotionSamples >= stationaryExitCount) {
                isStationary = false
            }
        }

        // Apply stationary override immediately
        if (isStationary) {
            currentSpeedMs = 0.0
            filteredGnssSpeed = 0.0
        }
    }

    /**
     * Ingests, validates, and filters incoming GNSS speed measurement.
     */
    fun updateGNSS(sample: GNSSSample) {
        val rawSpeed = sample.speedMs
        rawGnssSpeed = rawSpeed
        val t = sample.timestamp

        if (rawSpeed == null || rawSpeed.isNaN() || rawSpeed.isInfinite() || rawSpeed < 0.0 || rawSpeed > 80.0) {
            return
        }

        val dt = if (lastGnssTimestamp != null) {
            val delta = t - lastGnssTimestamp!!
            if (delta in 0.01..5.0) delta else 1.0
        } else {
            1.0
        }
        lastGnssTimestamp = t

        // 1. If IMU confirms stationary, reject GNSS speed noise and force zero
        if (isStationary) {
            filteredGnssSpeed = 0.0
            return
        }

        // 2. Low-speed deadband
        val validatedSpeed = if (rawSpeed < gnssDeadbandMs) 0.0 else rawSpeed

        // 3. Spike Protection / Max acceleration rate limiting relative to previous GNSS-filtered speed
        val baselineSpeed = filteredGnssSpeed
        val maxDeltaV = maxPlausibleAccel * dt
        val deltaV = validatedSpeed - baselineSpeed
        val clampedDeltaV = deltaV.coerceIn(-maxDeltaV, maxDeltaV)
        val rateLimitedSpeed = baselineSpeed + clampedDeltaV

        // 4. Exponential smoothing filter
        val alpha = 0.6
        filteredGnssSpeed = alpha * rateLimitedSpeed + (1.0 - alpha) * baselineSpeed
    }

    /**
     * Updates AI predicted forward speed and computes unified fused speed.
     * Keeps GNSS speed authoritative whenever a valid reliable GNSS speed is available.
     */
    fun updateAISpeed(predictedSpeed: Double, gnssState: GNSSState, isBlackout: Boolean) {
        aiPredictedSpeed = maxOf(0.0, if (predictedSpeed.isNaN() || predictedSpeed.isInfinite()) 0.0 else predictedSpeed)

        if (isStationary) {
            currentSpeedMs = 0.0
            return
        }

        val isGnssReliable = !isBlackout && (gnssState == GNSSState.AVAILABLE || gnssState == GNSSState.REACQUIRED)

        currentSpeedMs = if (isGnssReliable && filteredGnssSpeed > 0.1) {
            0.7 * filteredGnssSpeed + 0.3 * aiPredictedSpeed
        } else {
            // During GNSS outage / lost or cold start, AI model drives speed
            aiPredictedSpeed
        }
    }

    /**
     * Updates speed during cold-start period before full AI window is available.
     */
    fun updateColdStart(sample: IMUSample, dt: Double, gnssState: GNSSState, isBlackout: Boolean) {
        if (isStationary) {
            currentSpeedMs = 0.0
            return
        }

        val isGnssReliable = !isBlackout && (gnssState == GNSSState.AVAILABLE || gnssState == GNSSState.REACQUIRED)
        if (isGnssReliable && filteredGnssSpeed > 0.0) {
            currentSpeedMs = filteredGnssSpeed
        } else {
            // Damped integration of forward acceleration only when genuinely moving
            val ax = sample.ax
            if (abs(ax) > 0.3) {
                currentSpeedMs = maxOf(0.0, currentSpeedMs + ax * dt * 0.5)
            }
        }
    }

    /**
     * Returns diagnostic metrics for telemetry HUD and logging.
     */
    fun getDiagnostics(): Map<String, Any> {
        return mapOf(
            "stationary_detected" to isStationary,
            "raw_gnss_speed" to (rawGnssSpeed ?: 0.0),
            "filtered_gnss_speed" to filteredGnssSpeed,
            "ai_predicted_speed" to aiPredictedSpeed,
            "fused_speed" to currentSpeedMs
        )
    }
}
