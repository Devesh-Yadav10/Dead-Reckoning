package com.navigate.app

import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.NavigationState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NavigationStateTest {

    @Test
    fun testSpeedAndHeadingConversion() {
        val state = NavigationState(
            timestamp = 10.0,
            latitude = 51.5074,
            longitude = -0.1278,
            posEastM = 10.0,
            posNorthM = 20.0,
            speedMs = 10.0, // 10 m/s = 36.0 km/h
            headingDeg = 45.0,
            gnssState = GNSSState.AVAILABLE,
            blackoutActive = false
        )

        assertEquals(36.0, state.speedKmh, 1e-3)
        assertEquals("36.0 km/h", state.formattedSpeed)
        assertEquals("045°", state.formattedHeading)
        assertFalse(state.blackoutActive)
    }

    @Test
    fun testBlackoutStateAttributes() {
        val state = NavigationState(
            timestamp = 15.0,
            latitude = 51.5080,
            longitude = -0.1270,
            posEastM = 50.0,
            posNorthM = 100.0,
            speedMs = 15.0,
            headingDeg = 90.0,
            gnssState = GNSSState.BLACKOUT,
            blackoutActive = true,
            roadConstraintActive = true,
            matchedWayId = 12345L,
            matchedRoadName = "Kingsway Corridor"
        )

        assertTrue(state.blackoutActive)
        assertTrue(state.roadConstraintActive)
        assertEquals(GNSSState.BLACKOUT, state.gnssState)
        assertEquals(12345L, state.matchedWayId)
        assertEquals("Kingsway Corridor", state.matchedRoadName)
    }

    @Test
    fun testGNSSSampleCreation() {
        val sample = GNSSSample(
            timestamp = 1234.56,
            latitude = 51.5074,
            longitude = -0.1278,
            altitude = 15.2,
            accuracyM = 1.8
        )

        assertEquals(51.5074, sample.latitude, 1e-6)
        assertEquals(-0.1278, sample.longitude, 1e-6)
        assertEquals(1.8, sample.accuracyM, 1e-3)
    }
}

