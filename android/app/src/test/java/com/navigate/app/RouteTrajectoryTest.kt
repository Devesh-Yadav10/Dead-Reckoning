package com.navigate.app

import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.bridge.RouteTrajectoryGenerator
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import com.navigate.app.models.Route
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.osmdroid.util.GeoPoint
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class RouteTrajectoryTest {

    private fun createTestRoute(points: List<Pair<Double, Double>>): Route {
        val geoPoints = points.map { GeoPoint(it.first, it.second) }
        return Route(
            geometry = geoPoints,
            distanceMeters = 1000.0,
            durationSeconds = 100.0
        )
    }

    @Test
    fun testCumulativeDistanceCalculation() {
        // 3 points moving strictly North: (51.5000, 0.0) -> (51.5010, 0.0) -> (51.5020, 0.0)
        // 0.001 deg Lat ≈ 111.195 meters
        val route = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5010 to 0.0,
            51.5020 to 0.0
        ))

        val generator = RouteTrajectoryGenerator(route)
        assertTrue(generator.isValid)
        assertEquals(222.39, generator.totalDistanceMeters, 1.0)
    }

    @Test
    fun testMidSegmentInterpolation() {
        // 2 points: (51.5000, 0.0) to (51.5020, 0.0)
        val route = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5020 to 0.0
        ))

        val generator = RouteTrajectoryGenerator(route, initialLat = 51.5000, initialLon = 0.0)
        val halfDist = generator.totalDistanceMeters / 2.0

        val pt = generator.advance(halfDist, currentSpeedMs = 10.0, dt = 0.1)
        assertEquals(51.5010, pt.latitude, 1e-4)
        assertEquals(0.0, pt.longitude, 1e-5)
        assertEquals(0.5, pt.progress, 1e-2)
        assertFalse(pt.isCompleted)
    }

    @Test
    fun testNearestPointInitialization() {
        // Route from (51.5000, 0.0) to (51.5040, 0.0)
        val route = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5040 to 0.0
        ))

        // Vehicle starts at midpoint (51.5020, 0.0)
        val generator = RouteTrajectoryGenerator(
            route = route,
            initialLat = 51.5020,
            initialLon = 0.0,
            initialHeadingDeg = 0.0
        )

        // Travelled distance should be initialized close to half the route distance
        val expectedDist = generator.totalDistanceMeters / 2.0
        assertEquals(expectedDist, generator.travelledDistanceMeters, 2.0)

        // Next advance of 10m should advance from midpoint
        generator.advance(10.0, currentSpeedMs = 10.0, dt = 0.1)
        assertEquals(expectedDist + 10.0, generator.travelledDistanceMeters, 2.0)
    }

    @Test
    fun testHeadingCalculationAlongCardinalDirections() {
        // North leg -> East leg -> South leg -> West leg
        // Starting at (51.5000, 0.0)
        val route = createTestRoute(listOf(
            51.5000 to 0.0000,
            51.5010 to 0.0000, // North: 0°
            51.5010 to 0.0020, // East: 90°
            51.5000 to 0.0020, // South: 180°
            51.5000 to 0.0000  // West: 270°
        ))

        val generator = RouteTrajectoryGenerator(route)
        val pt1 = generator.sampleAtDistance(10.0) // On North leg
        assertEquals(0.0, pt1.headingDeg, 1.0)

        // Segment 2 distance (East leg)
        val pt2 = generator.sampleAtDistance(120.0)
        assertEquals(90.0, pt2.headingDeg, 1.0)
    }

    @Test
    fun testHeadingAngleWrapAroundShortestPath() {
        // Segment 1: Heading ~355°, Segment 2: Heading ~5° (Difference is +10°, not -350°)
        val route = createTestRoute(listOf(
            51.5000 to 0.0000,
            51.5010 to -0.0001, // ~355°
            51.5020 to 0.0001   // ~5°
        ))

        val generator = RouteTrajectoryGenerator(route, initialHeadingDeg = 355.0)
        val pt = generator.advance(150.0, currentSpeedMs = 10.0, dt = 0.1, maxTurnRateDegPerSec = 90.0)
        assertTrue("Heading should smoothly wrap around 0/360 boundary", pt.headingDeg in 0.0..15.0 || pt.headingDeg in 345.0..360.0)
    }

    @Test
    fun testHeadingSmoothingAcrossSharpTurns() {
        // Sharp 90 degree turn from North (0°) to East (90°)
        val route = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5005 to 0.0,   // North (0°) ~55m
            51.5005 to 0.0010  // East (90°) ~70m
        ))

        val generator = RouteTrajectoryGenerator(route, initialLat = 51.5000, initialLon = 0.0, initialHeadingDeg = 0.0)
        // Advance past corner with small dt (max turn = 90 deg/s * 0.1s = 9 deg per step)
        generator.advance(60.0, currentSpeedMs = 10.0, dt = 0.1, maxTurnRateDegPerSec = 90.0)
        // Heading should not instantly jump to 90°, but be bounded
        assertTrue("Heading should smoothly turn towards 90°", generator.currentHeadingDeg > 0.0 && generator.currentHeadingDeg <= 10.0)
    }

    @Test
    fun testRouteCompletionAndSpeedZeroing() {
        val route = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5005 to 0.0
        ))

        val generator = RouteTrajectoryGenerator(route)
        val total = generator.totalDistanceMeters
        val pt = generator.advance(total + 10.0, currentSpeedMs = 11.4, dt = 0.1)

        assertTrue(pt.isCompleted)
        assertEquals(1.0, pt.progress, 1e-6)
        assertEquals(total, generator.travelledDistanceMeters, 1e-4)
    }

    @Test
    fun testInvalidAndDegradedRoutes() {
        // Empty geometry
        val emptyRoute = Route(emptyList(), 0.0, 0.0)
        val gen1 = RouteTrajectoryGenerator(emptyRoute)
        assertFalse(gen1.isValid)
        val pt1 = gen1.advance(10.0)
        assertTrue(pt1.isCompleted)

        // Single point geometry
        val singlePtRoute = Route(listOf(GeoPoint(51.5, -0.1)), 0.0, 0.0)
        val gen2 = RouteTrajectoryGenerator(singlePtRoute)
        assertFalse(gen2.isValid)

        // NaN coordinates
        val nanRoute = Route(listOf(GeoPoint(Double.NaN, 0.0), GeoPoint(51.5, 0.0)), 0.0, 0.0)
        val gen3 = RouteTrajectoryGenerator(nanRoute)
        assertFalse(gen3.isValid)
    }

    @Test
    fun testMockBridgeFallbackWithoutRoute() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        assertNull(bridge.activeRoute)
        assertNull(bridge.routeTrajectoryGenerator)

        // Process IMU step
        bridge.processIMU(IMUSample(0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertNotNull(latestState)
        assertTrue(latestState!!.posEastM > 0.0)
        assertTrue(latestState!!.posNorthM > 0.0)
    }

    @Test
    fun testMockBridgeWithRouteActive() {
        val bridge = MockNavigationBridge()
        val route = createTestRoute(listOf(
            51.5074 to -0.1278,
            51.5090 to -0.1278,
            51.5090 to -0.1250
        ))

        bridge.startSession(51.5074, -0.1278, 0.0)
        bridge.setRoute(route)

        assertNotNull(bridge.activeRoute)
        assertNotNull(bridge.routeTrajectoryGenerator)
        assertTrue(bridge.routeTrajectoryGenerator!!.isValid)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        assertNotNull(latestState)
        assertTrue("Vehicle should move along the route geometry", latestState!!.latitude > 51.5074)
        assertEquals(-0.1278, latestState!!.longitude, 1e-4) // Still on north leg
    }

    @Test
    fun testBlackoutDoesNotDisruptRouteFollowing() {
        val bridge = MockNavigationBridge()
        val route = createTestRoute(listOf(
            51.5074 to -0.1278,
            51.5100 to -0.1278
        ))

        bridge.startSession(51.5074, -0.1278, 0.0)
        bridge.setRoute(route)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        // 1. Initial driving
        for (i in 1..5) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        val preBlackoutLat = latestState!!.latitude

        // 2. Trigger GNSS Outage
        bridge.setBlackout(true)

        // 3. Drive during outage
        for (i in 6..10) {
            bridge.processIMU(IMUSample(i * 0.1, 0.5, 0.0, 9.81, 0.0, 0.0, 0.0))
            assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
            assertTrue("Latitude must continue advancing along route during blackout", latestState!!.latitude > preBlackoutLat)
        }
        assertTrue("Matched road name should reflect OSM route", latestState!!.matchedRoadName?.startsWith("OSM Route") == true)

        // 4. Restore GNSS
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)
        assertNotNull(bridge.activeRoute)
    }

    @Test
    fun testRouteReplacementMidSession() {
        val bridge = MockNavigationBridge()
        val route1 = createTestRoute(listOf(
            51.5000 to 0.0,
            51.5050 to 0.0
        ))

        bridge.startSession(51.5000, 0.0, 0.0)
        bridge.setRoute(route1)

        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        val midLat = bridge.routeTrajectoryGenerator!!.sampleAtDistance(bridge.routeTrajectoryGenerator!!.travelledDistanceMeters).latitude

        // Replace with new route passing through the same area
        val route2 = createTestRoute(listOf(
            midLat to 0.0,
            51.5100 to 0.0
        ))

        bridge.setRoute(route2)
        assertNotNull(bridge.activeRoute)
        assertTrue(bridge.routeTrajectoryGenerator!!.isValid)

        // Trajectory should smoothly continue from near midLat without teleporting to (0,0)
        val newPt = bridge.routeTrajectoryGenerator!!.advance(5.0)
        assertTrue(newPt.latitude >= midLat)
    }

    @Test
    fun testClearDestinationRevertsToCorridor() {
        val bridge = MockNavigationBridge()
        val route = createTestRoute(listOf(
            51.5074 to -0.1278,
            51.5090 to -0.1278
        ))

        bridge.startSession(51.5074, -0.1278, 0.0)
        bridge.setRoute(route)
        assertNotNull(bridge.routeTrajectoryGenerator)

        // Clear route
        bridge.setRoute(null)
        assertNull(bridge.activeRoute)
        assertNull(bridge.routeTrajectoryGenerator)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        bridge.processIMU(IMUSample(0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertNotNull(latestState)
    }

    @Test
    fun testReplayModeAlongRoute() {
        val bridge = MockNavigationBridge()
        val route = createTestRoute(listOf(
            51.5074 to -0.1278,
            51.5100 to -0.1278
        ))

        bridge.setRoute(route)

        val latch = CountDownLatch(4)
        val states = mutableListOf<NavigationState>()

        bridge.startReplay { state ->
            states.add(state)
            latch.countDown()
        }

        val completed = latch.await(1000, TimeUnit.MILLISECONDS)
        bridge.stopReplay()

        assertTrue("Replay should emit states in real time", completed)
        assertTrue(states.size >= 4)
        for (st in states) {
            assertTrue("Latitude must be valid WGS84", st.latitude in 51.5074..51.5200)
            assertTrue("Speed must be finite", !st.speedMs.isNaN() && st.speedMs >= 0.0)
        }
    }
}
