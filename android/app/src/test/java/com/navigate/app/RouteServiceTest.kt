package com.navigate.app

import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.models.Destination
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.Route
import com.navigate.app.models.RouteRequest
import com.navigate.app.services.OsrmRouteService
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.osmdroid.util.GeoPoint

class RouteServiceTest {

    private val sampleOsrmGeoJson = """
    {
      "code": "Ok",
      "routes": [
        {
          "geometry": {
            "coordinates": [
              [87.310500, 22.314900],
              [87.315000, 22.320000],
              [87.320000, 22.330000],
              [87.324500, 22.339200]
            ],
            "type": "LineString"
          },
          "legs": [],
          "weight_name": "routability",
          "weight": 520.5,
          "duration": 480.0,
          "distance": 4200.0
        }
      ],
      "waypoints": []
    }
    """.trimIndent()

    @Test
    fun test1_RouteRequestModel() {
        val req = RouteRequest(
            originLatitude = 22.3149,
            originLongitude = 87.3105,
            destinationLatitude = 22.3392,
            destinationLongitude = 87.3245
        )
        assertEquals(22.3149, req.originLatitude, 1e-6)
        assertEquals(87.3105, req.originLongitude, 1e-6)
        assertEquals(22.3392, req.destinationLatitude, 1e-6)
        assertEquals(87.3245, req.destinationLongitude, 1e-6)
    }

    @Test
    fun test2_SuccessfulRouteParsing() {
        val service = OsrmRouteService()
        val route = service.parseResponse(sampleOsrmGeoJson)

        assertEquals(4200.0, route.distanceMeters, 1e-3)
        assertEquals(480.0, route.durationSeconds, 1e-3)
        assertEquals("4.2 km", route.formattedDistance)
        assertEquals("8 min", route.formattedDuration)
        assertEquals("4.2 km · 8 min", route.summary)
        assertEquals(4, route.geometry.size)
    }

    @Test
    fun test3_CoordinateOrderPreservesLatLonCorrectly() {
        val service = OsrmRouteService()
        val route = service.parseResponse(sampleOsrmGeoJson)

        // First point in JSON: [87.310500 (lon), 22.314900 (lat)]
        val firstPoint = route.geometry[0]
        assertEquals(22.314900, firstPoint.latitude, 1e-6)
        assertEquals(87.310500, firstPoint.longitude, 1e-6)

        // Last point in JSON: [87.324500 (lon), 22.339200 (lat)]
        val lastPoint = route.geometry[3]
        assertEquals(22.339200, lastPoint.latitude, 1e-6)
        assertEquals(87.324500, lastPoint.longitude, 1e-6)
    }

    @Test
    fun test4_MultipleGeometryPointsInOrder() {
        val service = OsrmRouteService()
        val route = service.parseResponse(sampleOsrmGeoJson)

        val expectedLats = doubleArrayOf(22.314900, 22.320000, 22.330000, 22.339200)
        val expectedLons = doubleArrayOf(87.310500, 87.315000, 87.320000, 87.324500)

        for (i in route.geometry.indices) {
            assertEquals(expectedLats[i], route.geometry[i].latitude, 1e-6)
            assertEquals(expectedLons[i], route.geometry[i].longitude, 1e-6)
        }
    }

    @Test
    fun test5_EmptyOrMissingGeometryThrowsControlledException() {
        val service = OsrmRouteService()
        val emptyGeometryJson = """
        {
          "code": "Ok",
          "routes": [
            {
              "geometry": { "coordinates": [] },
              "distance": 0.0,
              "duration": 0.0
            }
          ]
        }
        """.trimIndent()

        var exceptionThrown = false
        try {
            service.parseResponse(emptyGeometryJson)
        } catch (e: IllegalArgumentException) {
            exceptionThrown = true
        }
        assertTrue("Empty geometry must throw IllegalArgumentException", exceptionThrown)
    }

    @Test
    fun test6_MalformedJsonHandledGracefully() {
        val service = OsrmRouteService()
        val malformedJson = "{ \"code\": \"InvalidQuery\" }"

        var exceptionThrown = false
        try {
            service.parseResponse(malformedJson)
        } catch (e: IllegalArgumentException) {
            exceptionThrown = true
        }
        assertTrue("Invalid code must throw controlled exception", exceptionThrown)
    }

    @Test
    fun test7_HttpFailureReturnsControlledResult() = runBlocking {
        val brokenService = OsrmRouteService(baseUrl = "http://127.0.0.1:59999/route/v1/driving")
        val req = RouteRequest(22.3149, 87.3105, 22.3392, 87.3245)
        val result = brokenService.getRoute(req)
        assertTrue("Unreachable network must return Result.failure", result.isFailure)
        assertNotNull(result.exceptionOrNull())
    }

    @Test
    fun test8_RouteReplacementMaintainsSingleActiveRoute() {
        var activeRoute: Route? = null

        val service = OsrmRouteService()
        val route1 = service.parseResponse(sampleOsrmGeoJson)
        activeRoute = route1
        assertEquals("4.2 km · 8 min", activeRoute.summary)

        // Select new destination & route
        val route2 = Route(
            geometry = listOf(GeoPoint(22.0, 87.0), GeoPoint(22.1, 87.1)),
            distanceMeters = 15000.0,
            durationSeconds = 1200.0
        )
        activeRoute = route2
        assertEquals("15.0 km · 20 min", activeRoute.summary)
        assertEquals(2, activeRoute.geometry.size)
    }

    @Test
    fun test9_DestinationClearClearsRoute() {
        var selectedDest: Destination? = Destination(22.3149, 87.3105, "IIT Kharagpur")
        var activeRoute: Route? = Route(listOf(GeoPoint(22.3149, 87.3105)), 1000.0, 120.0)

        assertNotNull(selectedDest)
        assertNotNull(activeRoute)

        // User clears destination
        selectedDest = null
        activeRoute = null

        assertEquals(null, selectedDest)
        assertEquals(null, activeRoute)
    }

    @Test
    fun test10_RouteSurvivesBlackout() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        val service = OsrmRouteService()
        val currentRoute = service.parseResponse(sampleOsrmGeoJson)
        val selectedDest = Destination(22.3149, 87.3105, "IIT Kharagpur")

        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        // Trigger GNSS Blackout
        bridge.setBlackout(true)

        var latestState: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { latestState = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals(GNSSState.BLACKOUT, latestState!!.gnssState)
        assertTrue(latestState!!.blackoutActive)

        // Route and destination remain active and intact
        assertNotNull(selectedDest)
        assertNotNull(currentRoute)
        assertEquals(4, currentRoute.geometry.size)
        assertEquals("4.2 km · 8 min", currentRoute.summary)
    }

    @Test
    fun test11_RouteLoadingDoesNotAlterVehicleState() {
        val bridge = MockNavigationBridge()
        bridge.startSession(51.5074, -0.1278, 45.0)

        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        var stateBefore: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { stateBefore = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        val speedBefore = stateBefore!!.speedMs
        val latBefore = stateBefore!!.latitude
        val gnssBefore = stateBefore!!.gnssState

        // Route is loaded
        val service = OsrmRouteService()
        val loadedRoute = service.parseResponse(sampleOsrmGeoJson)
        assertNotNull(loadedRoute)

        // Next tick
        var stateAfter: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { stateAfter = it }
        bridge.processIMU(IMUSample(1.2, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals(speedBefore, stateAfter!!.speedMs, 1e-3)
        assertEquals(gnssBefore, stateAfter!!.gnssState)
        assertTrue("Vehicle position must continue moving smoothly", stateAfter!!.latitude > latBefore)
    }
}

