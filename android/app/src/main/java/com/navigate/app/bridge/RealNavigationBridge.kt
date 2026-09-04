package com.navigate.app.bridge

import android.content.Context
import android.util.Log
import ai.onnxruntime.OrtEnvironment
import com.navigate.app.ml.AttitudeOnnxModel
import com.navigate.app.ml.ModelMetadata
import com.navigate.app.ml.VelocityOnnxModel
import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import java.util.ArrayDeque
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * RealNavigationBridge — Mobile AI Dead-Reckoning Bridge (Kotlin + ONNX Runtime Mobile).
 *
 * Architecture & Design Note:
 * ---------------------------
 * This bridge is the production on-device execution engine for Android, providing:
 *   1. 50-sample sliding window IMU ingestion at 10 Hz (5.0s buffer).
 *   2. ONNX Runtime Mobile inference for VelocityModel V2 (forward speed estimation).
 *   3. ONNX Runtime Mobile inference for AttitudeModel (attitude quaternion inference).
 *   4. Stride-based 1.0s AI inference cadence (every 10 samples @ 10 Hz), matching RealTimeNavigationEngine.
 *   5. 10 Hz gyro angular rate integration and kinematic position dead-reckoning.
 *   6. Complete GNSS lifecycle state machine: AVAILABLE, LOST, BLACKOUT, REACQUIRED.
 *   7. Transparent OSM status reporting (OSM road matching is active in Replay mode and Python core;
 *      in this on-device live path, roadConstraintActive remains false until an on-device road cache is linked).
 *
 * Note: The full 15-state Invariant Extended Kalman Filter (ErrorStateIEKFTracker) covariance
 * propagation and live Overpass OSM client reside in the Python reference core (`src/navigate/`).
 */
class RealNavigationBridge(
    private val velocityModel: VelocityOnnxModel,
    private val attitudeModel: AttitudeOnnxModel
) : NavigationBridge, AutoCloseable {

    private var stateListener: ((NavigationState) -> Unit)? = null

    // Reference Anchor (WGS-84 origin for ENU frame)
    private var refLat = 51.5074
    private var refLon = -0.1278

    // Navigation State
    private var curLat = refLat
    private var curLon = refLon
    private var curEastM = 0.0
    private var curNorthM = 0.0
    private var curSpeedMs = 0.0
    private var curHeadingDeg = 0.0
    private var lastTimestamp: Double? = null

    // Blackout & GNSS State Management
    private var isBlackout = false
    private var isSessionActive = false
    private var gnssState: GNSSState = GNSSState.AVAILABLE
    private var lastGnssTimestamp: Double? = null
    private var hadPreviousFix = false

    // Centralized Speed Estimator (ZVD, GNSS Spike Filter, AI Speed Fusion)
    private val speedEstimator = MobileSpeedEstimator()

    // Continuous Gravity & Orientation Compensator
    private val gravityCompensator = GravityCompensator()

    // 50-sample IMU sliding buffer: each element is FloatArray(6) [ax, ay, az, gx, gy, gz]
    private val windowSize = 50
    private val stride = 10
    private val imuBuffer = ArrayDeque<FloatArray>(windowSize)
    private var sampleCount = 0
    private var lastInferenceSampleIdx = -1

    // GNSS timeout threshold for flagging LOST state (seconds)
    private val gnssTimeoutS = 3.0

    override fun startSession(refLat: Double, refLon: Double, initHeadingDeg: Double) {
        this.refLat = refLat
        this.refLon = refLon
        this.curLat = refLat
        this.curLon = refLon
        this.curEastM = 0.0
        this.curNorthM = 0.0
        this.curHeadingDeg = initHeadingDeg
        this.curSpeedMs = 0.0
        this.lastTimestamp = null
        this.isBlackout = false
        this.gnssState = GNSSState.AVAILABLE
        this.lastGnssTimestamp = null
        this.hadPreviousFix = false
        this.speedEstimator.reset()
        this.gravityCompensator.reset()
        this.imuBuffer.clear()
        this.sampleCount = 0
        this.lastInferenceSampleIdx = -1
        this.isSessionActive = true
    }

    override fun stopSession() {
        isSessionActive = false
    }

    override fun reset() {
        stopSession()
        curEastM = 0.0
        curNorthM = 0.0
        curHeadingDeg = 0.0
        curSpeedMs = 0.0
        lastTimestamp = null
        isBlackout = false
        gnssState = GNSSState.AVAILABLE
        lastGnssTimestamp = null
        hadPreviousFix = false
        speedEstimator.reset()
        gravityCompensator.reset()
        imuBuffer.clear()
        sampleCount = 0
        lastInferenceSampleIdx = -1
    }

    override fun setBlackout(enabled: Boolean) {
        this.isBlackout = enabled
        if (this.isBlackout) {
            this.gnssState = GNSSState.BLACKOUT
        }
    }

    override fun setNavigationStateListener(listener: (NavigationState) -> Unit) {
        this.stateListener = listener
    }

    override fun processGNSS(sample: GNSSSample) {
        if (!isSessionActive) return
        lastGnssTimestamp = sample.timestamp

        // Pass GNSS speed to speed estimator for validation and spike filtering
        speedEstimator.updateGNSS(sample)
        curSpeedMs = speedEstimator.currentSpeedMs

        Log.d(
            "NAVIGATE_SPEED_DEBUG",
            "[GNSS Fix Processed] rawGnssSpeedMs=${sample.speedMs?.let { "%.2f (%.1f km/h)".format(it, it * 3.6) } ?: "null"}, " +
                    "filteredGnssSpeedMs=${"%.2f (%.1f km/h)".format(speedEstimator.filteredGnssSpeed, speedEstimator.filteredGnssSpeed * 3.6)}, " +
                    "fusedSpeedMs=${"%.2f (%.1f km/h)".format(curSpeedMs, curSpeedMs * 3.6)}, " +
                    "stationaryDetected=${speedEstimator.isStationary}, gnssState=$gnssState"
        )

        if (!isBlackout) {
            val wasOutage = (gnssState == GNSSState.BLACKOUT || gnssState == GNSSState.LOST)
            gnssState = if (wasOutage) GNSSState.REACQUIRED else GNSSState.AVAILABLE
            hadPreviousFix = true

            curLat = sample.latitude
            curLon = sample.longitude

            // Update ENU displacement relative to anchor
            val dLat = Math.toRadians(curLat - refLat)
            val dLon = Math.toRadians(curLon - refLon)
            curNorthM = dLat * 6371000.0
            curEastM = dLon * (6371000.0 * cos(Math.toRadians(refLat)))

            if (sample.bearingDeg != null && curSpeedMs > 1.5) {
                curHeadingDeg = sample.bearingDeg
            }
        }
    }

    override fun processIMU(sample: IMUSample) {
        if (!isSessionActive) return
        val t0 = System.nanoTime()

        val dt = if (lastTimestamp != null) {
            val delta = sample.timestamp - lastTimestamp!!
            if (delta in 0.001..0.500) delta else 0.100
        } else {
            0.100
        }
        lastTimestamp = sample.timestamp

        // 1. Gravity Estimation and Orientation Leveling
        val compResult = gravityCompensator.process(sample, dt)

        // 2. Update Stationary Detector (ZVD) with streaming IMU sample and linear acceleration magnitude
        speedEstimator.updateIMU(sample, compResult.linearAccelMag)

        // Add raw sample to sliding buffer (preserves exact raw phone-frame channels)
        val rawSample = floatArrayOf(
            sample.ax.toFloat(),
            sample.ay.toFloat(),
            sample.az.toFloat(),
            sample.gx.toFloat(),
            sample.gy.toFloat(),
            sample.gz.toFloat()
        )

        if (imuBuffer.size >= windowSize) {
            imuBuffer.removeFirst()
        }
        imuBuffer.addLast(rawSample)
        sampleCount++

        Log.d(
            "NAVIGATE_GRAVITY_DEBUG",
            "[GRAVITY_COMP] rawAccel=[%.3f, %.3f, %.3f], gravVec=[%.3f, %.3f, %.3f], linAccel=[%.3f, %.3f, %.3f] (|a_lin|=%.3f), compAccel=[%.3f, %.3f, %.3f], stationary=%b"
                .format(
                    sample.ax, sample.ay, sample.az,
                    compResult.gravityX, compResult.gravityY, compResult.gravityZ,
                    compResult.linearAx, compResult.linearAy, compResult.linearAz, compResult.linearAccelMag,
                    compResult.compAx, compResult.compAy, compResult.compAz,
                    speedEstimator.isStationary
                )
        )

        // 3. Check GNSS timeout for LOST state transition
        if (!isBlackout && hadPreviousFix && lastGnssTimestamp != null) {
            if ((sample.timestamp - lastGnssTimestamp!!) > gnssTimeoutS) {
                if (gnssState != GNSSState.LOST) {
                    gnssState = GNSSState.LOST
                }
            }
        }

        // 4. Attitude & Heading Update
        // Integrate gyro yaw rate at 10 Hz only if not completely stationary to avoid gyro bias drift
        if (!speedEstimator.isStationary) {
            curHeadingDeg = (curHeadingDeg + Math.toDegrees(sample.gz * dt)) % 360.0
            if (curHeadingDeg < 0.0) curHeadingDeg += 360.0
        }

        // 5. Stride-based AI Model Inference (1.0s Cadence = 10 samples @ 10 Hz)
        if (imuBuffer.size == windowSize) {
            val shouldInfer = (lastInferenceSampleIdx < 0) || ((sampleCount - lastInferenceSampleIdx) >= stride)

            if (shouldInfer) {
                val rawWindow = imuBuffer.toList()
                val leveledWindow = gravityCompensator.compensateWindow(rawWindow)

                // A. VelocityModel Inference (Forward Speed with gravity compensation)
                val predictedSpeed = velocityModel.predictSpeed(leveledWindow).toDouble()

                Log.d(
                    "NAVIGATE_SPEED_DEBUG",
                    "[ONNX Inference] AI_PRE_FUSION_SPEED=${"%.2f m/s (%.1f km/h)".format(predictedSpeed, predictedSpeed * 3.6)}, " +
                            "rawGnssSpeedMs=${speedEstimator.rawGnssSpeed?.let { "%.2f".format(it) } ?: "null"}, " +
                            "filteredGnssSpeedMs=${"%.2f".format(speedEstimator.filteredGnssSpeed)}, " +
                            "stationaryDetected=${speedEstimator.isStationary}, gnssState=$gnssState, " +
                            "linAccelMag=${"%.3f".format(compResult.linearAccelMag)}"
                )

                speedEstimator.updateAISpeed(predictedSpeed, gnssState, isBlackout)

                Log.d(
                    "NAVIGATE_SPEED_DEBUG",
                    "[ONNX Post-Fusion] fusedSpeedMs=${"%.2f m/s (%.1f km/h)".format(speedEstimator.currentSpeedMs, speedEstimator.currentSpeedMs * 3.6)}"
                )

                // B. AttitudeModel Inference (Quaternion Orientation)
                if (!speedEstimator.isStationary) {
                    val q = attitudeModel.predictQuaternion(rawWindow)
                    val qw = q[0].toDouble()
                    val qx = q[1].toDouble()
                    val qy = q[2].toDouble()
                    val qz = q[3].toDouble()
                    val sinyCosp = 2.0 * (qw * qz + qx * qy)
                    val cosyCosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                    val aiYawDeg = Math.toDegrees(atan2(sinyCosp, cosyCosp))

                    if (curSpeedMs > 1.0) {
                        val normalizedYaw = ((aiYawDeg % 360.0) + 360.0) % 360.0
                        curHeadingDeg = 0.95 * curHeadingDeg + 0.05 * normalizedYaw
                    }
                }

                lastInferenceSampleIdx = sampleCount
            }
        } else {
            // Cold-start prior to 50 samples
            speedEstimator.updateColdStart(sample, dt, gnssState, isBlackout)
        }

        // Update current speed from speed estimator
        curSpeedMs = speedEstimator.currentSpeedMs

        // 5. Kinematic Position Dead-Reckoning (Only integrate position when moving)
        if (curSpeedMs > 0.0) {
            val headingRad = Math.toRadians(curHeadingDeg)
            val dEast = curSpeedMs * sin(headingRad) * dt
            val dNorth = curSpeedMs * cos(headingRad) * dt

            curEastM += dEast
            curNorthM += dNorth

            // Convert ENU to WGS84 Geodetic Coordinates
            curLat = refLat + (curNorthM / 6371000.0) * (180.0 / Math.PI)
            curLon = refLon + (curEastM / (6371000.0 * cos(Math.toRadians(refLat)))) * (180.0 / Math.PI)
        }

        val t1 = System.nanoTime()
        val latencyMs = (t1 - t0) / 1_000_000.0

        // Combine base diagnostics with speed estimator diagnostics
        val diag = mutableMapOf<String, Any>(
            "buffer_size" to imuBuffer.size,
            "step" to sampleCount,
            "engine" to "ONNX Runtime Mobile"
        )
        diag.putAll(speedEstimator.getDiagnostics())

        // 6. Emit Truthful Navigation State
        val state = NavigationState(
            timestamp = sample.timestamp,
            latitude = curLat,
            longitude = curLon,
            posEastM = curEastM,
            posNorthM = curNorthM,
            speedMs = curSpeedMs,
            headingDeg = curHeadingDeg,
            gnssState = gnssState,
            blackoutActive = isBlackout,
            roadConstraintActive = false,
            matchedWayId = null,
            matchedRoadName = null,
            latencyMs = latencyMs,
            diagnostics = diag
        )

        stateListener?.invoke(state)
    }

    override fun close() {
        velocityModel.close()
        attitudeModel.close()
    }

    companion object {
        /**
         * Factory function to instantiate RealNavigationBridge from Android Context assets.
         */
        fun createFromAssets(context: Context): RealNavigationBridge {
            val env = OrtEnvironment.getEnvironment()

            val metadataStr = context.assets.open("model_metadata.json").use { it.bufferedReader().readText() }
            val metadata = ModelMetadata.fromJson(metadataStr)

            val velBytes = context.assets.open(metadata.velocityConfig.file).use { it.readBytes() }
            val attBytes = context.assets.open(metadata.attitudeConfig.file).use { it.readBytes() }

            val velModel = VelocityOnnxModel(env, velBytes, metadata.velocityConfig)
            val attModel = AttitudeOnnxModel(env, attBytes, metadata.attitudeConfig)

            return RealNavigationBridge(velModel, attModel)
        }
    }
}
