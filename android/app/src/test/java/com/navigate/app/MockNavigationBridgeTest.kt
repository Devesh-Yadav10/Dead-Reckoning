package com.navigate.app

import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class MockNavigationBridgeTest {

    @Test
    fun testLiveSampleProcessing() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 0.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { state ->
            latestState = state
        }

        // Send acceleration forward (ax = 2.0 m/s^2)
        val imu = IMUSample(
            timestamp = 0.100,
            ax = 2.0,
            ay = 0.0,
            az = 9.81,
            gx = 0.0,
            gy = 0.0,
            gz = 0.0
        )
        bridge.processIMU(imu)

        assertNotNull(latestState)
        assertEquals(0.100, latestState!!.timestamp, 1e-6)
        assertTrue(latestState!!.speedMs > 0.0)
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
    }

    @Test
    fun testBlackoutToggle() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 0.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { state ->
            latestState = state
        }

        bridge.setBlackout(true)
        bridge.processIMU(IMUSample(0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertNotNull(latestState)
        assertTrue(latestState!!.blackoutActive)
        assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
        assertTrue(latestState!!.roadConstraintActive)
    }

    @Test
    fun testReplayModeProgression() {
        val bridge = MockNavigationBridge()
        val latch = CountDownLatch(3)
        val receivedStates = mutableListOf<NavigationState>()

        bridge.startReplay { state ->
            receivedStates.add(state)
            latch.countDown()
        }

        val completed = latch.await(1000, TimeUnit.MILLISECONDS)
        bridge.stopReplay()

        assertTrue("Replay should emit states in real time", completed)
        assertTrue(receivedStates.size >= 3)
    }

    @Test
    fun testInteractiveBlackoutTransitionsAndReacquisition() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 0.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { state ->
            latestState = state
        }

        // 1. Initial State -> AVAILABLE
        bridge.processIMU(IMUSample(0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertNotNull(latestState)
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
        assertEquals(false, latestState!!.blackoutActive)

        // 2. User toggles Outage Simulation ON -> BLACKOUT
        bridge.setBlackout(true)
        bridge.processIMU(IMUSample(0.2, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
        assertEquals(true, latestState!!.blackoutActive)
        assertEquals(true, latestState!!.roadConstraintActive)
        assertTrue(latestState!!.speedMs > 0.0)

        // 3. User clicks RESTORE GNSS -> REACQUIRED
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(0.3, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)
        assertEquals(false, latestState!!.blackoutActive)

        // 4. Exhaust 20 reacquisition ticks (2.0s at 10Hz) -> AVAILABLE
        for (i in 4..25) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
        assertEquals(false, latestState!!.blackoutActive)
    }

    @Test
    fun testDeterministicTrajectory() {
        val bridge1 = MockNavigationBridge()
        val bridge2 = MockNavigationBridge()
        bridge1.startSession(51.5074, -0.1278, 45.0)
        bridge2.startSession(51.5074, -0.1278, 45.0)

        val states1 = mutableListOf<NavigationState>()
        val states2 = mutableListOf<NavigationState>()

        bridge1.setNavigationStateListener { states1.add(it) }
        bridge2.setNavigationStateListener { states2.add(it) }

        for (i in 1..50) {
            val imu = IMUSample(i * 0.1, 1.5, 0.0, 9.81, 0.0, 0.0, 0.05)
            bridge1.processIMU(imu)
            bridge2.processIMU(imu)
        }

        assertEquals(50, states1.size)
        assertEquals(50, states2.size)

        for (i in 0 until 50) {
            val s1 = states1[i]
            val s2 = states2[i]
            assertEquals("Latitudes must match at step $i", s1.latitude, s2.latitude, 1e-12)
            assertEquals("Longitudes must match at step $i", s1.longitude, s2.longitude, 1e-12)
            assertEquals("East pos must match at step $i", s1.posEastM, s2.posEastM, 1e-12)
            assertEquals("North pos must match at step $i", s1.posNorthM, s2.posNorthM, 1e-12)
            assertEquals("Speed must match at step $i", s1.speedMs, s2.speedMs, 1e-12)
            assertEquals("Heading must match at step $i", s1.headingDeg, s2.headingDeg, 1e-12)
        }
    }

    @Test
    fun testBlackoutDoesNotFreezePositionMovement() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        // 1. Accelerate
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        val preBlackoutEast = latestState!!.posEastM
        val preBlackoutNorth = latestState!!.posNorthM
        val speed = latestState!!.speedMs
        assertTrue(speed > 0.5)

        // 2. Trigger Blackout
        bridge.setBlackout(true)

        // 3. Process ticks during blackout
        for (i in 11..20) {
            val prevEast = latestState!!.posEastM
            val prevNorth = latestState!!.posNorthM
            bridge.processIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

            // Position must continue moving during blackout
            assertTrue("East position must advance during blackout at tick $i", latestState!!.posEastM > prevEast)
            assertTrue("North position must advance during blackout at tick $i", latestState!!.posNorthM > prevNorth)
            assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
            assertTrue(latestState!!.blackoutActive)
        }

        assertTrue("Cumulative East position must be significantly greater than pre-blackout", latestState!!.posEastM > preBlackoutEast + 0.5)
        assertTrue("Cumulative North position must be significantly greater than pre-blackout", latestState!!.posNorthM > preBlackoutNorth + 0.5)
    }

    @Test
    fun testRestoreDoesNotResetPosition() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        // Enter Blackout
        bridge.setBlackout(true)
        for (i in 11..20) {
            bridge.processIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        val blackoutEndEast = latestState!!.posEastM
        val blackoutEndNorth = latestState!!.posNorthM
        val blackoutEndLat = latestState!!.latitude
        val blackoutEndLon = latestState!!.longitude

        // Restore GNSS
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(2.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        // Position MUST NOT jump back to origin / starting fix
        assertTrue("East position must not reset to 0", latestState!!.posEastM >= blackoutEndEast)
        assertTrue("North position must not reset to 0", latestState!!.posNorthM >= blackoutEndNorth)
        assertTrue("Latitude must be continuous", latestState!!.latitude >= blackoutEndLat)
        assertTrue("Longitude must be continuous", latestState!!.longitude >= blackoutEndLon)
        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)
    }

    @Test
    fun testRepeatedBlackoutRestoreCycles() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        var prevEast = 0.0
        var t = 0.1

        // 3 sequential blackout / restore cycles
        for (cycle in 1..3) {
            // Normal phase
            bridge.setBlackout(false)
            for (step in 1..5) {
                bridge.processIMU(IMUSample(t, 0.5, 0.0, 9.81, 0.0, 0.0, 0.0))
                assertTrue("East pos must monotonically advance", latestState!!.posEastM >= prevEast)
                prevEast = latestState!!.posEastM
                t += 0.1
            }

            // Blackout phase
            bridge.setBlackout(true)
            for (step in 1..5) {
                bridge.processIMU(IMUSample(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
                assertTrue("East pos must monotonically advance during blackout", latestState!!.posEastM > prevEast)
                assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
                prevEast = latestState!!.posEastM
                t += 0.1
            }

            // Reacquired phase
            bridge.setBlackout(false)
            bridge.processIMU(IMUSample(t, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
            assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)
            assertTrue("East pos must not reset", latestState!!.posEastM >= prevEast)
            prevEast = latestState!!.posEastM
            t += 0.1
        }
    }
}

