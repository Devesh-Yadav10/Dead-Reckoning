package com.navigate.app

import com.navigate.app.bridge.MobileSpeedEstimator
import com.navigate.app.bridge.RealNavigationBridge
import com.navigate.app.ml.AttitudeOnnxModel
import com.navigate.app.ml.ModelMetadata
import com.navigate.app.ml.VelocityOnnxModel
import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.Mockito

class RealNavigationBridgeTest {

    private val jsonMetadata = """
    {
      "velocity_model": {
        "file": "velocity_model_v2.onnx",
        "input_name": "imu_input",
        "output_name": "velocity_output",
        "imu_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "imu_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "target_mean": 11.5,
        "target_std": 8.6
      },
      "attitude_model": {
        "file": "attitude_model.onnx",
        "input_name": "imu_input",
        "output_name": "attitude_output",
        "imu_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "imu_std": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      }
    }
    """.trimIndent()

    @Test
    fun testModelMetadataParsing() {
        val meta = ModelMetadata.fromJson(jsonMetadata)
        assertEquals("velocity_model_v2.onnx", meta.velocityConfig.file)
        assertEquals("attitude_model.onnx", meta.attitudeConfig.file)
        assertEquals(11.5f, meta.velocityConfig.targetMean, 1e-4f)
        assertEquals(8.6f, meta.velocityConfig.targetStd, 1e-4f)
        assertEquals(6, meta.velocityConfig.imuMean.size)
        assertEquals(6, meta.attitudeConfig.imuMean.size)
    }

    @Test
    fun testStationaryPhoneWithZeroGnssSpeed() {
        // Requirement 7.A: Stationary phone with GNSS speed = 0 -> speed = 0.0
        val estimator = MobileSpeedEstimator()
        // Feed 10 stationary IMU samples (at rest: gravity on Z, no rotation)
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 0.0))
        estimator.updateAISpeed(0.0, GNSSState.AVAILABLE, isBlackout = false)

        assertTrue(estimator.isStationary)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
    }

    @Test
    fun testStationaryPhoneWithGnssSpeedSpike() {
        // Requirement 7.B: Stationary phone with 15.6 km/h (4.33 m/s) GNSS spike -> speed remains 0.0
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        // GNSS spike of 4.33 m/s (15.6 km/h)
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 4.33))

        assertTrue(estimator.isStationary)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
        assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)
    }

    @Test
    fun testMultipleStationaryGnssSpikes() {
        // Requirement 7.C: Multiple stationary GNSS spikes -> speed remains 0.0
        val estimator = MobileSpeedEstimator()
        for (i in 1..20) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
            if (i % 5 == 0) {
                // Random multipath spikes
                estimator.updateGNSS(GNSSSample(i * 0.1, 51.5074, -0.1278, speedMs = (i % 3 + 2).toDouble()))
            }
        }
        assertTrue(estimator.isStationary)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
    }

    @Test
    fun testGenuineLowSpeedMotion() {
        // Requirement 7.D: Genuine low-speed motion (e.g., 1.5 m/s) with dynamic IMU is not clamped to 0
        val estimator = MobileSpeedEstimator()
        // Feed moving IMU samples (rotational energy and dynamic acceleration)
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.8, 0.2, 10.6, 0.15, 0.05, 0.20))
        }
        assertFalse(estimator.isStationary)

        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 1.5))
        estimator.updateAISpeed(1.2, GNSSState.AVAILABLE, isBlackout = false)

        assertTrue(estimator.currentSpeedMs > 0.5)
    }

    @Test
    fun testGenuineAcceleration() {
        // Requirement 7.E: Genuine acceleration allows speed to increase smoothly
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 2.5, 0.0, 10.5, 0.2, 0.0, 0.0))
        }
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 5.0))
        estimator.updateAISpeed(4.8, GNSSState.AVAILABLE, isBlackout = false)

        assertTrue(estimator.currentSpeedMs > 2.0)
    }

    @Test
    fun testAiPredictsNonzeroWhileStationary() {
        // Requirement 7.F: AI predicts nonzero speed (e.g., 3.5 m/s) while stationary -> ZVD forces speed to 0.0
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        estimator.updateAISpeed(3.5, GNSSState.AVAILABLE, isBlackout = false)

        assertTrue(estimator.isStationary)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
    }

    @Test
    fun testGnssLostWhileMovingUsesAiSpeed() {
        // Requirement 7.G: GNSS LOST while moving -> AI speed remains primary usable speed
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.2, 0.0, 10.8, 0.2, 0.0, 0.0))
        }
        estimator.updateAISpeed(12.5, GNSSState.LOST, isBlackout = false)

        assertFalse(estimator.isStationary)
        assertEquals(12.5, estimator.currentSpeedMs, 1e-6)
    }

    @Test
    fun testGnssReacquisitionSmoothTransition() {
        // Requirement 7.H: GNSS reacquisition smoothly transitions speed
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.0, 0.0, 10.5, 0.15, 0.0, 0.0))
        }
        // During blackout, AI predicted 10.0 m/s
        estimator.updateAISpeed(10.0, GNSSState.BLACKOUT, isBlackout = true)
        assertEquals(10.0, estimator.currentSpeedMs, 1e-6)

        // GNSS reacquired at 11.0 m/s
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 11.0))
        estimator.updateAISpeed(10.0, GNSSState.REACQUIRED, isBlackout = false)

        assertTrue(estimator.currentSpeedMs in 5.0..12.0)
    }

    @Test
    fun testSessionReset() {
        // Requirement 7.I: Reset clears all estimator state
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 2.0, 0.0, 11.0, 0.3, 0.0, 0.0))
        }
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 15.0))
        assertFalse(estimator.isStationary)

        estimator.reset()
        assertTrue(estimator.isStationary)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
        assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)
    }

    @Test
    fun testAiCannotSeedGnssFilter() {
        // TEST 1: Initial filtered GNSS = 0, current fused/AI speed = 21.65 m/s.
        // Inject GNSS raw speed = 0.0 m/s.
        // Assert filtered GNSS is NOT 17-18 m/s (must remain 0.0 m/s).
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.0, 0.0, 10.5, 0.15, 0.0, 0.0))
        }
        assertFalse(estimator.isStationary)
        assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)

        // AI sets fused speed to 21.65 m/s
        estimator.updateAISpeed(21.65, GNSSState.AVAILABLE, isBlackout = false)
        assertEquals(21.65, estimator.currentSpeedMs, 1e-6)

        // Inject raw GNSS = 0.0 m/s
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 0.0))

        // Filtered GNSS must NOT be corrupted by AI speed (must be 0.0, NOT ~17-18 m/s)
        assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)
    }

    @Test
    fun testRepeatedZeroGnssDoesNotCreateArtificialSpeed() {
        // TEST 2: Set AI speed to large value (20 m/s). Feed multiple GNSS fixes: 0.0, 0.0, 0.0, 0.0.
        // Assert filtered GNSS remains 0.0 rather than producing artificial decay sequence.
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.0, 0.0, 10.5, 0.15, 0.0, 0.0))
        }
        assertFalse(estimator.isStationary)

        estimator.updateAISpeed(20.0, GNSSState.AVAILABLE, isBlackout = false)

        for (sec in 1..5) {
            estimator.updateGNSS(GNSSSample(sec.toDouble(), 51.5074, -0.1278, speedMs = 0.0))
            assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)
        }
    }

    @Test
    fun testGnssFilterDoesNotConsumeFusedSpeed() {
        // TEST 3: GNSS = 5 m/s, AI changes dramatically (2 m/s -> 20 m/s -> 0 m/s).
        // Filtered GNSS state should depend only on GNSS inputs/history.
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.0, 0.0, 10.5, 0.15, 0.0, 0.0))
        }
        assertFalse(estimator.isStationary)

        // GNSS fix at 5.0 m/s
        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 5.0))
        val initialFilteredGnss = estimator.filteredGnssSpeed
        assertTrue(initialFilteredGnss > 0.0)

        // AI changes dramatically
        estimator.updateAISpeed(2.0, GNSSState.AVAILABLE, isBlackout = false)
        assertEquals(initialFilteredGnss, estimator.filteredGnssSpeed, 1e-6)

        estimator.updateAISpeed(20.0, GNSSState.AVAILABLE, isBlackout = false)
        assertEquals(initialFilteredGnss, estimator.filteredGnssSpeed, 1e-6)

        estimator.updateAISpeed(0.0, GNSSState.AVAILABLE, isBlackout = false)
        assertEquals(initialFilteredGnss, estimator.filteredGnssSpeed, 1e-6)
    }

    @Test
    fun testPreserveLegitimateGnssSmoothing() {
        // TEST 4: Realistic non-zero GNSS measurements smoothly ramp up according to rate limiting & smoothing.
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.5, 0.0, 10.8, 0.2, 0.0, 0.0))
        }
        assertFalse(estimator.isStationary)

        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 5.0))
        val speed1 = estimator.filteredGnssSpeed
        assertTrue(speed1 in 2.5..3.5) // alpha * 5.0 = 3.0 m/s

        estimator.updateGNSS(GNSSSample(2.0, 51.5074, -0.1278, speedMs = 8.0))
        val speed2 = estimator.filteredGnssSpeed
        assertTrue(speed2 > speed1)
        assertTrue(speed2 in 5.5..7.0)

        estimator.updateGNSS(GNSSSample(3.0, 51.5074, -0.1278, speedMs = 8.0))
        val speed3 = estimator.filteredGnssSpeed
        assertTrue(speed3 > speed2)
        assertTrue(speed3 in 7.0..8.0)
    }

    @Test
    fun testFusionUsesBothSourcesDownstream() {
        // TEST 5: Filtered GNSS remains independent, while fused speed incorporates AI (0.7 GNSS + 0.3 AI).
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 1.0, 0.0, 10.5, 0.15, 0.0, 0.0))
        }
        assertFalse(estimator.isStationary)

        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 10.0))
        val gnssVal = estimator.filteredGnssSpeed

        val aiVal = 8.0
        estimator.updateAISpeed(aiVal, GNSSState.AVAILABLE, isBlackout = false)

        val expectedFused = 0.7 * gnssVal + 0.3 * aiVal
        assertEquals(expectedFused, estimator.currentSpeedMs, 1e-4)
    }

    @Test
    fun testStationaryBehaviorRemainsIntact() {
        // TEST 6: When stationary detection is true, fused/displayed speed and filtered GNSS are forced to 0.0.
        val estimator = MobileSpeedEstimator()
        for (i in 1..10) {
            estimator.updateIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        assertTrue(estimator.isStationary)

        estimator.updateGNSS(GNSSSample(1.0, 51.5074, -0.1278, speedMs = 5.0))
        estimator.updateAISpeed(10.0, GNSSState.AVAILABLE, isBlackout = false)

        assertEquals(0.0, estimator.filteredGnssSpeed, 1e-6)
        assertEquals(0.0, estimator.currentSpeedMs, 1e-6)
    }

    @Test
    fun testRealBridgeDynamicsAndGNSSStateTransitions() {
        val mockVel = Mockito.mock(VelocityOnnxModel::class.java)
        val mockAtt = Mockito.mock(AttitudeOnnxModel::class.java)

        Mockito.`when`(mockVel.predictSpeed(Mockito.anyList())).thenReturn(10.0f)
        Mockito.`when`(mockAtt.predictQuaternion(Mockito.anyList())).thenReturn(floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f))

        val bridge = RealNavigationBridge(mockVel, mockAtt)
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 0.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { state ->
            latestState = state
        }

        // 1. Initial fix
        bridge.processGNSS(GNSSSample(0.0, 51.5074, -0.1278, 10.0, 2.0, 10.0, 45.0))

        // 2. Feed 50 moving samples to fill the window (with 1 Hz GNSS fixes at integer seconds)
        for (i in 1..50) {
            val t = i * 0.100
            if (i % 10 == 0) {
                bridge.processGNSS(GNSSSample(t, 51.5074, -0.1278, 10.0, 2.0, 10.0, 45.0))
            }
            // Dynamic motion: total acceleration deviates from 1g, non-zero angular rate
            val imu = IMUSample(
                timestamp = t,
                ax = 1.2,
                ay = 0.0,
                az = 10.8,
                gx = 0.15,
                gy = 0.0,
                gz = 0.0
            )
            bridge.processIMU(imu)
        }

        assertNotNull(latestState)
        assertEquals(5.0, latestState!!.timestamp, 1e-6)
        assertTrue(latestState!!.speedMs > 0.0)
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
        assertFalse(latestState!!.blackoutActive)
        assertFalse(latestState!!.roadConstraintActive)
        assertNull(latestState!!.matchedWayId)
        assertNull(latestState!!.matchedRoadName)

        // 3. Test Manual Blackout state
        bridge.setBlackout(true)
        bridge.processIMU(IMUSample(5.1, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))

        assertNotNull(latestState)
        assertTrue(latestState!!.blackoutActive)
        assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)

        // 4. Test GNSS Reacquisition (BLACKOUT -> REACQUIRED -> AVAILABLE)
        bridge.setBlackout(false)
        bridge.processGNSS(GNSSSample(5.2, 51.5076, -0.1276, 10.0, 2.0, 10.0, 45.0))
        bridge.processIMU(IMUSample(5.2, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))

        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)
        assertFalse(latestState!!.blackoutActive)

        // Subsequent fix transitions REACQUIRED -> AVAILABLE
        bridge.processGNSS(GNSSSample(5.3, 51.5077, -0.1275, 10.0, 2.0, 10.0, 45.0))
        bridge.processIMU(IMUSample(5.3, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)

        // 5. Test GNSS Timeout -> LOST transition (> 3.0s without GNSS fix)
        bridge.processIMU(IMUSample(8.5, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))
        assertEquals(GNSSState.LOST, latestState!!.gnssState)

        // 6. Test Recovery from LOST (LOST -> REACQUIRED -> AVAILABLE)
        bridge.processGNSS(GNSSSample(8.6, 51.5080, -0.1270, 10.0, 2.0, 10.0, 45.0))
        bridge.processIMU(IMUSample(8.6, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))
        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)

        bridge.processGNSS(GNSSSample(8.7, 51.5081, -0.1269, 10.0, 2.0, 10.0, 45.0))
        bridge.processIMU(IMUSample(8.7, 1.2, 0.0, 10.8, 0.15, 0.0, 0.0))
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)

        bridge.close()
    }

    @Test
    fun testMLInferenceReceivesRawPhoneFrameWindow() {
        val mockVel = Mockito.mock(VelocityOnnxModel::class.java)
        val mockAtt = Mockito.mock(AttitudeOnnxModel::class.java)

        Mockito.`when`(mockVel.predictSpeed(Mockito.anyList())).thenReturn(0.54f)
        Mockito.`when`(mockAtt.predictQuaternion(Mockito.anyList())).thenReturn(floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f))

        val bridge = RealNavigationBridge(mockVel, mockAtt)
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 0.0)

        // Feed 50 identical asymmetric samples: ax=1.11, ay=2.22, az=3.33, gx=4.44, gy=5.55, gz=6.66
        for (i in 1..50) {
            val t = i * 0.100
            val sample = IMUSample(
                timestamp = t,
                ax = 1.11,
                ay = 2.22,
                az = 3.33,
                gx = 4.44,
                gy = 5.55,
                gz = 6.66
            )
            bridge.processIMU(sample)
        }

        // Verify that the internal buffer has 50 samples with exact asymmetric channels
        val bufferField = RealNavigationBridge::class.java.getDeclaredField("imuBuffer")
        bufferField.isAccessible = true
        @Suppress("UNCHECKED_CAST")
        val buffer = bufferField.get(bridge) as java.util.ArrayDeque<FloatArray>

        assertEquals(50, buffer.size)
        val firstSample = buffer.first()
        assertEquals(6, firstSample.size)
        // Ensure channels are strictly [ax, ay, az, gx, gy, gz] and NOT permuted [ay, -ax, az, gy, -gx, gz]
        assertEquals(1.11f, firstSample[0], 1e-4f)  // ax
        assertEquals(2.22f, firstSample[1], 1e-4f)  // ay
        assertEquals(3.33f, firstSample[2], 1e-4f)  // az
        assertEquals(4.44f, firstSample[3], 1e-4f)  // gx
        assertEquals(5.55f, firstSample[4], 1e-4f)  // gy
        assertEquals(6.66f, firstSample[5], 1e-4f)  // gz

        bridge.close()
    }
}
