package com.navigate.app

import com.navigate.app.bridge.GravityCompensator
import com.navigate.app.bridge.MobileSpeedEstimator
import com.navigate.app.bridge.RealNavigationBridge
import com.navigate.app.ml.AttitudeOnnxModel
import com.navigate.app.ml.VelocityOnnxModel
import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.Mockito
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

class GravityCompensationTest {

    @Test
    fun testStationaryPhoneFlat() {
        val compensator = GravityCompensator()
        // Flat on table: ax=0, ay=0, az=9.81, gyro=0
        val sample = IMUSample(0.1, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        val res = compensator.process(sample)

        assertEquals(0.0, res.linearAx, 1e-3)
        assertEquals(0.0, res.linearAy, 1e-3)
        assertEquals(0.0, res.linearAz, 1e-3)
        assertEquals(0.0, res.linearAccelMag, 1e-3)
        assertEquals(0.0, res.compAx, 1e-3)
        assertEquals(0.0, res.compAy, 1e-3)
        assertEquals(9.80665, res.compAz, 1e-3)
    }

    @Test
    fun testStationaryPhoneTilted90Degrees() {
        val compensator = GravityCompensator()
        // Tilted 90 degrees in portrait: ax=0, ay=9.80665, az=0, gyro=0
        val sample = IMUSample(0.1, 0.0, 9.80665, 0.0, 0.0, 0.0, 0.0)
        val res = compensator.process(sample)

        // Estimated gravity in body should align with Y
        assertEquals(0.0, res.gravityX, 1e-2)
        assertEquals(9.80665, res.gravityY, 1e-2)
        assertEquals(0.0, res.gravityZ, 1e-2)

        // Linear acceleration must be zero
        assertEquals(0.0, res.linearAx, 1e-3)
        assertEquals(0.0, res.linearAy, 1e-3)
        assertEquals(0.0, res.linearAz, 1e-3)
        assertEquals(0.0, res.linearAccelMag, 1e-3)

        // Compensated acceleration leveled to nominal reference frame: [0, 0, 9.80665]
        assertEquals(0.0, res.compAx, 1e-3)
        assertEquals(0.0, res.compAy, 1e-3)
        assertEquals(9.80665, res.compAz, 1e-3)
    }

    @Test
    fun testStationaryPhoneRotatedAroundVerticalAxis() {
        val compensator = GravityCompensator()
        // Flat phone rotated at yaw rate gz = 0.5 rad/s: ax=0, ay=0, az=9.80665
        for (i in 1..20) {
            val sample = IMUSample(i * 0.1, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.5)
            val res = compensator.process(sample, dt = 0.1)

            // Linear acceleration remains ~0
            assertTrue("Linear accel magnitude must remain near 0 during vertical rotation", res.linearAccelMag < 0.05)
            assertEquals(9.80665, res.gravityZ, 1e-2)
            assertEquals(0.0, res.compAx, 1e-2)
            assertEquals(0.0, res.compAy, 1e-2)
            assertEquals(9.80665, res.compAz, 1e-2)
        }
    }

    @Test
    fun testStationaryPhoneUndergoingOrientationChanges() {
        val compensator = GravityCompensator()
        val estimator = MobileSpeedEstimator()

        // Slowly rotate pitch from 0 to 45 degrees over 2 seconds (pitch rate ~0.39 rad/s)
        val totalSteps = 20
        for (i in 1..totalSteps) {
            val t = i * 0.100
            val pitchRad = Math.toRadians(45.0 * (i.toDouble() / totalSteps))
            val pitchRate = Math.toRadians(45.0 / 2.0) // ~0.3927 rad/s

            val ax = 0.0
            val ay = 9.80665 * sin(pitchRad)
            val az = 9.80665 * cos(pitchRad)
            val gx = pitchRate
            val gy = 0.0
            val gz = 0.0

            val sample = IMUSample(t, ax, ay, az, gx, gy, gz)
            val res = compensator.process(sample, dt = 0.100)
            estimator.updateIMU(sample, res.linearAccelMag)

            // Linear acceleration must remain small (no translational acceleration)
            assertTrue("Linear acceleration must remain bounded during rotation, got ${res.linearAccelMag}", res.linearAccelMag < 0.5)
            // Compensated forward accel must not falsely register gravity as high forward acceleration
            assertTrue("Compensated ay must remain small, got ${res.compAy}", abs(res.compAy) < 1.0)
        }
    }

    @Test
    fun testGenuineTranslationalAcceleration() {
        val compensator = GravityCompensator()

        // 1. Initial rest
        val rest = IMUSample(0.1, 0.0, 0.0, 9.80665, 0.0, 0.0, 0.0)
        compensator.process(rest)

        // 2. Forward vehicle acceleration of 2.5 m/s^2 along body +Y (ay = 2.5, az = 9.80665)
        val movingSample = IMUSample(0.2, 0.0, 2.5, 9.80665, 0.0, 0.0, 0.0)
        val res = compensator.process(movingSample)

        // Linear acceleration must reflect the 2.5 m/s^2 translational acceleration
        assertEquals(0.0, res.linearAx, 1e-2)
        assertEquals(2.5, res.linearAy, 1e-2)
        assertEquals(0.0, res.linearAz, 1e-2)
        assertEquals(2.5, res.linearAccelMag, 1e-2)

        // Compensated acceleration must preserve the 2.5 m/s^2 forward component
        assertEquals(2.5, res.compAy, 1e-2)
        assertEquals(9.80665, res.compAz, 1e-2)
    }

    @Test
    fun testGnssSpeedOverridingUnreliableAiSpeedWhenAppropriate() {
        val estimator = MobileSpeedEstimator()

        // 1. Phone moving: IMU has motion
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 1.5, 9.80665, 0.05, 0.0, 0.0), linearAccelMag = 1.5)
        }
        assertFalse(estimator.isStationary)

        // 2. Converge reliable GNSS speed with consecutive 1 Hz fixes
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 10.0))
        estimator.updateGNSS(GNSSSample(2.0, 51.5074, -0.1278, speedMs = 10.0))
        estimator.updateGNSS(GNSSSample(3.0, 51.5074, -0.1278, speedMs = 10.0))
        estimator.updateGNSS(GNSSSample(4.0, 51.5074, -0.1278, speedMs = 10.0))

        // 3. AI model predicts 25.0 m/s (e.g. noisy estimate)
        estimator.updateAISpeed(25.0, GNSSState.AVAILABLE, isBlackout = false)

        // Fused speed is pulled strongly towards GNSS speed (0.7 * ~10.0 + 0.3 * 25.0 = ~14.5 m/s vs 25.0 m/s)
        assertTrue(estimator.currentSpeedMs < 16.0)
        assertTrue(estimator.currentSpeedMs >= 10.0)
    }

    @Test
    fun testRealBridgeTiltedStationaryPhoneProducesZeroSpeed() {
        val mockVel = Mockito.mock(VelocityOnnxModel::class.java)
        val mockAtt = Mockito.mock(AttitudeOnnxModel::class.java)

        // When leveled window is passed (compAy ~ 0, compAz ~ 9.81), velocity model returns 0.0
        Mockito.`when`(mockVel.predictSpeed(Mockito.anyList())).thenReturn(0.0f)
        Mockito.`when`(mockAtt.predictQuaternion(Mockito.anyList())).thenReturn(floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f))

        val bridge = RealNavigationBridge(mockVel, mockAtt)
        bridge.startSession(refLat = 28.618790, refLon = 76.964707, initHeadingDeg = 0.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        // Feed 50 samples of tilted phone: ay=9.371, az=2.662 (total mag = 9.74 m/s^2, stationary)
        for (i in 1..50) {
            val t = i * 0.100
            val sample = IMUSample(
                timestamp = t,
                ax = 0.383,
                ay = 9.371,
                az = 2.662,
                gx = 0.01,
                gy = 0.01,
                gz = 0.01
            )
            bridge.processIMU(sample)
        }

        assertNotNull(latestState)
        assertEquals(0.0, latestState!!.speedMs, 1e-4)
        bridge.close()
    }
}
