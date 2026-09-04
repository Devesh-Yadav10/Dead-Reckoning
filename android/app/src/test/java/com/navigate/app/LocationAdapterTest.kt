package com.navigate.app

import android.content.Context
import android.location.Location
import com.google.android.gms.location.FusedLocationProviderClient
import com.navigate.app.models.GNSSSample
import com.navigate.app.sensors.AndroidLocationAdapter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.Mockito

class LocationAdapterTest {

    private fun createMockLocation(
        lat: Double,
        lon: Double,
        alt: Double = 0.0,
        accuracy: Float = 5.0f,
        speed: Float? = null,
        bearing: Float? = null,
        time: Long = 1700000000000L,
        provider: String = "gps"
    ): Location {
        val loc = Mockito.mock(Location::class.java)
        Mockito.`when`(loc.latitude).thenReturn(lat)
        Mockito.`when`(loc.longitude).thenReturn(lon)
        Mockito.`when`(loc.time).thenReturn(time)
        Mockito.`when`(loc.provider).thenReturn(provider)

        if (alt != 0.0) {
            Mockito.`when`(loc.hasAltitude()).thenReturn(true)
            Mockito.`when`(loc.altitude).thenReturn(alt)
        } else {
            Mockito.`when`(loc.hasAltitude()).thenReturn(false)
        }

        Mockito.`when`(loc.hasAccuracy()).thenReturn(true)
        Mockito.`when`(loc.accuracy).thenReturn(accuracy)

        if (speed != null) {
            Mockito.`when`(loc.hasSpeed()).thenReturn(true)
            Mockito.`when`(loc.speed).thenReturn(speed)
        } else {
            Mockito.`when`(loc.hasSpeed()).thenReturn(false)
        }

        if (bearing != null) {
            Mockito.`when`(loc.hasBearing()).thenReturn(true)
            Mockito.`when`(loc.bearing).thenReturn(bearing)
        } else {
            Mockito.`when`(loc.hasBearing()).thenReturn(false)
        }

        return loc
    }

    @Test
    fun testLocationConversionPreservesAllFields() {
        val mockContext = Mockito.mock(Context::class.java)
        var capturedSample: GNSSSample? = null

        val adapter = AndroidLocationAdapter(mockContext) { sample ->
            capturedSample = sample
        }

        val location = createMockLocation(
            lat = 28.618790,
            lon = 76.964707,
            alt = 215.5,
            accuracy = 4.2f,
            speed = 12.5f,
            bearing = 135.0f,
            time = 1715000000000L,
            provider = "fused"
        )

        adapter.onLocationChanged(location)

        assertNotNull(capturedSample)
        assertEquals(1715000000.0, capturedSample!!.timestamp, 1e-3)
        assertEquals(28.618790, capturedSample!!.latitude, 1e-6)
        assertEquals(76.964707, capturedSample!!.longitude, 1e-6)
        assertEquals(215.5, capturedSample!!.altitude, 1e-3)
        assertEquals(4.2, capturedSample!!.accuracyM, 1e-3)
        assertEquals(12.5, capturedSample!!.speedMs!!, 1e-3)
        assertEquals(135.0, capturedSample!!.bearingDeg!!, 1e-3)
    }

    @Test
    fun testFirstLiveFixInitializesRealLocation() {
        val mockContext = Mockito.mock(Context::class.java)
        val samples = mutableListOf<GNSSSample>()

        val adapter = AndroidLocationAdapter(mockContext) { sample ->
            samples.add(sample)
        }

        // Real Delhi physical fix coordinates
        val firstFix = createMockLocation(lat = 28.618790, lon = 76.964707)
        adapter.onLocationChanged(firstFix)

        assertEquals(1, samples.size)
        assertEquals(28.618790, samples[0].latitude, 1e-6)
        assertEquals(76.964707, samples[0].longitude, 1e-6)
        assertTrue(adapter.hasReceivedFreshFix)
    }

    @Test
    fun testSubsequentFixesUpdatePosition() {
        val mockContext = Mockito.mock(Context::class.java)
        val samples = mutableListOf<GNSSSample>()

        val adapter = AndroidLocationAdapter(mockContext) { sample ->
            samples.add(sample)
        }

        val fix1 = createMockLocation(lat = 28.618790, lon = 76.964707)
        val fix2 = createMockLocation(lat = 28.618850, lon = 76.964780)
        val fix3 = createMockLocation(lat = 28.618910, lon = 76.964850)

        adapter.onLocationChanged(fix1)
        adapter.onLocationChanged(fix2)
        adapter.onLocationChanged(fix3)

        assertEquals(3, samples.size)
        assertEquals(28.618790, samples[0].latitude, 1e-6)
        assertEquals(28.618850, samples[1].latitude, 1e-6)
        assertEquals(28.618910, samples[2].latitude, 1e-6)
    }

    @Test
    fun testStaleLastKnownLocationDoesNotOverwriteFreshFix() {
        val mockContext = Mockito.mock(Context::class.java)
        val samples = mutableListOf<GNSSSample>()

        val adapter = AndroidLocationAdapter(mockContext) { sample ->
            samples.add(sample)
        }

        // 1. Fresh live fix arrives first
        val freshFix = createMockLocation(lat = 28.618790, lon = 76.964707)
        adapter.processLocation(freshFix, isLastKnown = false)

        assertEquals(1, samples.size)
        assertTrue(adapter.hasReceivedFreshFix)

        // 2. Delayed stale last-known location arrives afterwards (e.g. old London cache)
        val staleLastKnown = createMockLocation(lat = 51.5074, lon = -0.1278)
        adapter.processLocation(staleLastKnown, isLastKnown = true)

        // Must be discarded; sample list still size 1 with Delhi coordinates
        assertEquals(1, samples.size)
        assertEquals(28.618790, samples[0].latitude, 1e-6)
        assertEquals(76.964707, samples[0].longitude, 1e-6)
    }

    @Test
    fun testValidLastKnownLocationServesAsInitialFallback() {
        val mockContext = Mockito.mock(Context::class.java)
        val samples = mutableListOf<GNSSSample>()

        val adapter = AndroidLocationAdapter(mockContext) { sample ->
            samples.add(sample)
        }

        // Last known location arrives before any fresh fix
        val lastKnown = createMockLocation(lat = 28.618790, lon = 76.964707)
        adapter.processLocation(lastKnown, isLastKnown = true)

        assertEquals(1, samples.size)
        assertEquals(28.618790, samples[0].latitude, 1e-6)
        assertFalse(adapter.hasReceivedFreshFix) // Still waiting for continuous fresh fix

        // Subsequent fresh fix replaces/updates it
        val freshFix = createMockLocation(lat = 28.618800, lon = 76.964710)
        adapter.processLocation(freshFix, isLastKnown = false)

        assertEquals(2, samples.size)
        assertEquals(28.618800, samples[1].latitude, 1e-6)
        assertTrue(adapter.hasReceivedFreshFix)
    }

    @Test
    fun testPermissionOrProviderFailureHandling() {
        val mockContext = Mockito.mock(Context::class.java)
        val mockClient = Mockito.mock(FusedLocationProviderClient::class.java)
        var caughtError: Throwable? = null

        Mockito.`when`(mockClient.lastLocation).thenThrow(SecurityException("ACCESS_FINE_LOCATION denied"))

        val adapter = AndroidLocationAdapter(
            context = mockContext,
            fusedLocationClient = mockClient,
            onError = { caughtError = it }
        ) { }

        adapter.startListening()

        assertFalse(adapter.isListening)
        assertNotNull(caughtError)
        assertTrue(caughtError is SecurityException)
    }
}

