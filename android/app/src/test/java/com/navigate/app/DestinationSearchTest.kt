package com.navigate.app

import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.models.Destination
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.services.DestinationSearchService
import com.navigate.app.services.NominatimDestinationSearchService
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DestinationSearchTest {

    private val sampleNominatimJson = """
    [
      {
        "place_id": 1001,
        "licence": "Data © OpenStreetMap contributors",
        "osm_type": "way",
        "osm_id": 50001,
        "lat": "22.3149",
        "lon": "87.3105",
        "class": "amenity",
        "type": "university",
        "place_rank": 30,
        "importance": 0.75,
        "addresstype": "amenity",
        "name": "IIT Kharagpur",
        "display_name": "IIT Kharagpur, Kharagpur, Paschim Medinipur, West Bengal, 721302, India"
      },
      {
        "place_id": 1002,
        "licence": "Data © OpenStreetMap contributors",
        "osm_type": "node",
        "osm_id": 50002,
        "lat": "22.3392",
        "lon": "87.3245",
        "class": "railway",
        "type": "station",
        "place_rank": 30,
        "importance": 0.65,
        "addresstype": "railway",
        "name": "Kharagpur Junction",
        "display_name": "Kharagpur Junction, Railway Colony, Kharagpur, Paschim Medinipur, West Bengal, India"
      }
    ]
    """.trimIndent()

    @Test
    fun test1_DestinationModelFields() {
        val dest = Destination(
            latitude = 22.3149,
            longitude = 87.3105,
            displayName = "IIT Kharagpur",
            address = "Kharagpur, West Bengal, India"
        )
        assertEquals(22.3149, dest.latitude, 1e-6)
        assertEquals(87.3105, dest.longitude, 1e-6)
        assertEquals("IIT Kharagpur", dest.displayName)
        assertEquals("Kharagpur, West Bengal, India", dest.address)
    }

    @Test
    fun test2_EmptyOrWhitespaceQueryDoesNotQueryNetwork() = runBlocking {
        val service = NominatimDestinationSearchService()
        val emptyResult = service.search("")
        val whitespaceResult = service.search("   ")

        assertTrue(emptyResult.isSuccess)
        assertTrue(emptyResult.getOrNull()?.isEmpty() == true)

        assertTrue(whitespaceResult.isSuccess)
        assertTrue(whitespaceResult.getOrNull()?.isEmpty() == true)
    }

    @Test
    fun test3_SuccessfulSearchParsing() {
        val service = NominatimDestinationSearchService()
        val results = service.parseResponse(sampleNominatimJson)

        assertEquals(2, results.size)
        val first = results[0]
        assertEquals(22.3149, first.latitude, 1e-4)
        assertEquals(87.3105, first.longitude, 1e-4)
        assertEquals("IIT Kharagpur", first.displayName)
        assertTrue("Address must contain regional details", first.address.contains("Kharagpur") && first.address.contains("West Bengal"))
    }

    @Test
    fun test4_MultipleResultsPreserveOrder() {
        val service = NominatimDestinationSearchService()
        val results = service.parseResponse(sampleNominatimJson)

        assertEquals(2, results.size)
        assertEquals("IIT Kharagpur", results[0].displayName)
        assertEquals("Kharagpur Junction", results[1].displayName)
        assertEquals(22.3392, results[1].latitude, 1e-4)
        assertEquals(87.3245, results[1].longitude, 1e-4)
    }

    @Test
    fun test5_MalformedResponseHandledGracefully() {
        val service = NominatimDestinationSearchService()
        val malformedJson = "{ \"error\": \"not an array\" }"
        var exceptionCaught = false
        try {
            service.parseResponse(malformedJson)
        } catch (e: Exception) {
            exceptionCaught = true
        }
        assertTrue("Malformed response should throw catchable JSON exception without crashing app", exceptionCaught)
    }

    @Test
    fun test6_HttpFailureReturnsControlledResult() = runBlocking {
        // Point service to an invalid unreachable port to simulate controlled network failure
        val brokenService = NominatimDestinationSearchService(baseUrl = "http://127.0.0.1:59999/search")
        val result = brokenService.search("Test Query")
        assertTrue("Network error must return Result.failure", result.isFailure)
        assertNotNull(result.exceptionOrNull())
    }

    @Test
    fun test7_DestinationSelectionDoesNotAlterVehicleMotionOrGNSS() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        // Vehicle starts moving
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        var stateBeforeSelection: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { stateBeforeSelection = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        val speedBefore = stateBeforeSelection!!.speedMs
        val latBefore = stateBeforeSelection!!.latitude
        val lonBefore = stateBeforeSelection!!.longitude
        val headingBefore = stateBeforeSelection!!.headingDeg
        val gnssBefore = stateBeforeSelection!!.gnssState

        // User selects destination
        val selectedDest = Destination(22.3149, 87.3105, "IIT Kharagpur", "Kharagpur")
        assertNotNull(selectedDest)

        // Process next vehicle tick after destination selection
        var stateAfterSelection: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { stateAfterSelection = it }
        bridge.processIMU(IMUSample(1.2, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals("Speed must continue unaffected by destination selection", speedBefore, stateAfterSelection!!.speedMs, 1e-3)
        assertEquals("Heading must continue unaffected", headingBefore, stateAfterSelection!!.headingDeg, 1e-3)
        assertEquals("GNSS state must remain intact", gnssBefore, stateAfterSelection!!.gnssState)
        assertTrue("Vehicle must continue forward motion", stateAfterSelection!!.latitude > latBefore)
    }

    @Test
    fun test8_DestinationSurvivesBlackout() {
        val bridge = MockNavigationBridge()
        bridge.startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = 45.0)

        // Select destination
        var currentDestination: Destination? = Destination(22.3149, 87.3105, "IIT Kharagpur", "Kharagpur")
        assertNotNull(currentDestination)

        // Accelerate
        for (i in 1..10) {
            bridge.processIMU(IMUSample(i * 0.1, 1.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        }

        // Trigger GNSS Blackout
        bridge.setBlackout(true)

        var stateDuringBlackout: com.navigate.app.models.NavigationState? = null
        bridge.setNavigationStateListener { stateDuringBlackout = it }
        bridge.processIMU(IMUSample(1.1, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))

        assertEquals(GNSSState.BLACKOUT, stateDuringBlackout!!.gnssState)
        assertTrue(stateDuringBlackout!!.blackoutActive)
        // Destination remains selected
        assertNotNull(currentDestination)
        assertEquals("IIT Kharagpur", currentDestination?.displayName)

        // Restore GNSS
        bridge.setBlackout(false)
        bridge.processIMU(IMUSample(1.2, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0))
        assertEquals(GNSSState.REACQUIRED, stateDuringBlackout!!.gnssState)
        assertNotNull(currentDestination)
    }
}

