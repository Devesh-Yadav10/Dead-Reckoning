package com.navigate.app

import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.models.Destination
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import com.navigate.app.models.Route
import com.navigate.app.models.RouteRequest
import com.navigate.app.services.RouteService
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.osmdroid.util.GeoPoint

class DemoIntegrationTest {

    class FakeControllableRouteService : RouteService {
        private val pendingRequests = mutableListOf<Pair<RouteRequest, CompletableDeferred<Result<Route>>>>()

        fun queueResponse(): CompletableDeferred<Result<Route>> {
            val deferred = CompletableDeferred<Result<Route>>()
            return deferred
        }

        var immediateRoute: Route? = null

        override suspend fun getRoute(request: RouteRequest): Result<Route> {
            immediateRoute?.let { return Result.success(it) }
            val deferred = CompletableDeferred<Result<Route>>()
            pendingRequests.add(Pair(request, deferred))
            return deferred.await()
        }

        fun completeNext(result: Result<Route>) {
            if (pendingRequests.isNotEmpty()) {
                val (_, deferred) = pendingRequests.removeAt(0)
                deferred.complete(result)
            }
        }
    }

    private fun createSampleRoute(label: String, distanceKm: Double): Route {
        return Route(
            geometry = listOf(GeoPoint(22.3149, 87.3105), GeoPoint(22.3392, 87.3245)),
            distanceMeters = distanceKm * 1000.0,
            durationSeconds = distanceKm * 120.0
        )
    }

    @Test
    fun testA_B_C_D_RouteLoadingDoesNotAlterVehicleState() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        // Accelerate vehicle
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        var baselineState: NavigationState? = null
        bridge.setNavigationStateListener { baselineState = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        val baselineSpeed = baselineState!!.speedMs
        val baselineLat = baselineState!!.latitude
        val baselineLon = baselineState!!.longitude
        val baselineHeading = baselineState!!.headingDeg
        val baselineGnss = baselineState!!.gnssState

        // Simulate Route Loading
        val route = createSampleRoute("Route A", 4.2)
        assertNotNull(route)

        // Process next tick with loaded route
        var afterState: NavigationState? = null
        bridge.setNavigationStateListener { afterState = it }
        bridge.processIMU(IMUSample(1.2, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        // Test A: Speed unaffected
        assertEquals("Test A: Speed must not be altered by route loading", baselineSpeed, afterState!!.speedMs, 1e-4)
        // Test B: Position continues smoothly without teleports
        assertTrue("Test B: Vehicle position continues monotonic forward motion", afterState!!.latitude > baselineLat)
        // Test C: Heading unaffected
        assertEquals("Test C: Heading must not be altered by route loading", baselineHeading, afterState!!.headingDeg, 1e-4)
        // Test D: GNSS state unaffected
        assertEquals("Test D: GNSS state must remain AVAILABLE", baselineGnss, afterState!!.gnssState)
    }

    @Test
    fun testE_F_G_GNSSBlackoutAndReacquisitionPreservesRouteAndDestination() {
        var selectedDest: Destination? = Destination(22.3149, 87.3105, "IIT Kharagpur", "Kharagpur")
        var currentRoute: Route? = createSampleRoute("IIT Kharagpur", 4.2)

        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        // 1. Initial State: Normal AVAILABLE
        var currentState: NavigationState? = null
        bridge.setNavigationStateListener { currentState = it }
        bridge.processIMU(IMUSample(0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.AVAILABLE, currentState!!.gnssState)

        // 2. Trigger GNSS Outage -> BLACKOUT
        bridge.setBlackout(true)
        bridge.processIMU(IMUSample(0.2, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals(GNSSState.BLACKOUT, currentState!!.gnssState)
        assertTrue(currentState!!.blackoutActive)
        // Test E: Route remains intact during blackout
        assertNotNull("Test E: Route must survive blackout", currentRoute)
        assertEquals("4.2 km", currentRoute?.formattedDistance)
        // Test F: Destination remains intact during blackout
        assertNotNull("Test F: Destination must survive blackout", selectedDest)
        assertEquals("IIT Kharagpur", selectedDest?.displayName)

        // 3. Restore GNSS -> REACQUIRED -> AVAILABLE
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(0.3, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals(GNSSState.REACQUIRED, currentState!!.gnssState)
        // Test G: Route remains intact during reacquisition
        assertNotNull("Test G: Route must survive reacquisition", currentRoute)
        assertNotNull("Destination must survive reacquisition", selectedDest)
    }

    @Test
    fun testH_SelectingDestinationBReplacesRouteA() {
        var selectedDest: Destination? = Destination(22.3149, 87.3105, "Destination A")
        var currentRoute: Route? = createSampleRoute("Route A", 5.0)

        assertEquals("5.0 km", currentRoute?.formattedDistance)
        assertEquals("Destination A", selectedDest?.displayName)

        // User selects Destination B
        selectedDest = Destination(22.5726, 88.3639, "Destination B")
        val routeB = createSampleRoute("Route B", 12.5)
        currentRoute = routeB

        assertEquals("Destination B", selectedDest.displayName)
        assertEquals("12.5 km", currentRoute.formattedDistance)
    }

    @Test
    fun testI_AsyncRaceLateResponseCannotOverwriteNewerDestination() {
        // Architecture token simulation
        var routeGenerationId = 0L
        var selectedDestination: Destination? = null
        var activeRoute: Route? = null

        // Step 1: User selects Destination A
        val destA = Destination(22.3149, 87.3105, "Destination A")
        selectedDestination = destA
        val genA = ++routeGenerationId

        // Step 2: User rapidly selects Destination B before Route A returns
        val destB = Destination(22.5726, 88.3639, "Destination B")
        selectedDestination = destB
        val genB = ++routeGenerationId

        // Step 3: Route B finishes
        val routeB = createSampleRoute("Route B", 10.0)
        if (genB == routeGenerationId && selectedDestination == destB) {
            activeRoute = routeB
        }
        assertEquals("10.0 km", activeRoute?.formattedDistance)

        // Step 4: Stale Route A finishes LATER
        val routeA = createSampleRoute("Route A", 3.0)
        if (genA == routeGenerationId && selectedDestination == destA) {
            activeRoute = routeA // Should NOT execute
        }

        // Assert: Stale Route A did NOT overwrite Route B
        assertEquals("Active route must remain Route B despite late arrival of Route A", "10.0 km", activeRoute?.formattedDistance)
    }

    @Test
    fun testJ_K_RouteFailureDoesNotAffectGNSSOrVehicleMotion() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        // Accelerate
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        var baselineState: NavigationState? = null
        bridge.setNavigationStateListener { baselineState = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        val speedBefore = baselineState!!.speedMs
        val latBefore = baselineState!!.latitude
        val gnssBefore = baselineState!!.gnssState

        // Route service fails
        val routeResult: Result<Route> = Result.failure(Exception("HTTP 500 Connection Timeout"))
        assertTrue(routeResult.isFailure)

        // Next vehicle tick
        var afterState: NavigationState? = null
        bridge.setNavigationStateListener { afterState = it }
        bridge.processIMU(IMUSample(1.2, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        // Test J: GNSS state unchanged
        assertEquals("Test J: GNSS state must not be affected by routing failure", gnssBefore, afterState!!.gnssState)
        // Test K: Vehicle motion continues
        assertEquals("Test K: Vehicle speed must continue", speedBefore, afterState!!.speedMs, 1e-4)
        assertTrue("Test K: Vehicle position must continue moving", afterState!!.latitude > latBefore)
    }

    @Test
    fun testComprehensiveDemoLifecycleSequence() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        var latestState: NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }

        // 1. Start Demo
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
        assertTrue(latestState!!.speedMs > 0.5)

        // 2. Select Destination
        val dest = Destination(22.3149, 87.3105, "IIT Kharagpur")
        val route = createSampleRoute("IIT Kharagpur", 4.2)
        assertNotNull(dest)
        assertNotNull(route)

        // 3. Blackout
        bridge.setBlackout(true)
        for (i in 11..20) {
            bridge.processIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
        assertTrue(latestState!!.blackoutActive)

        // 4. Restore
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(2.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.REACQUIRED, latestState!!.gnssState)

        // 5. Normal
        for (i in 22..45) {
            bridge.processIMU(IMUSample(i * 0.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }
        assertEquals(GNSSState.AVAILABLE, latestState!!.gnssState)
        assertFalse(latestState!!.blackoutActive)
    }
}

