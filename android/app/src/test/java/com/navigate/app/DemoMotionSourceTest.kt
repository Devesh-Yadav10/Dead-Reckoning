package com.navigate.app

import com.navigate.app.bridge.DemoMotionSource
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class DemoMotionSourceTest {

    @Test
    fun test1_Determinism() {
        val source1 = DemoMotionSource()
        val source2 = DemoMotionSource()

        val steps = 300 // 30.0s @ 10Hz
        for (i in 0..steps) {
            val t = i * 0.100
            val speed1 = source1.calculateSpeedMs(t)
            val speed2 = source2.calculateSpeedMs(t)
            assertEquals("Speed at t=$t must be strictly deterministic", speed1, speed2, 0.0)
        }
    }

    @Test
    fun test2_StartsStationary() {
        val source = DemoMotionSource()

        // 0.0s to 2.0s must be exactly 0.0 km/h
        for (i in 0..20) {
            val t = i * 0.100
            val speedMs = source.calculateSpeedMs(t)
            val speedKmh = source.calculateSpeedKmh(t)
            assertEquals(0.0, speedMs, 1e-9)
            assertEquals(0.0, speedKmh, 1e-9)
        }
    }

    @Test
    fun test3_SmoothAcceleration() {
        val source = DemoMotionSource()

        var prevSpeed = 0.0
        val maxAllowedDeltaPer100ms = 0.35 // ~1.26 km/h per frame max allowable

        for (i in 21..80) { // 2.1s to 8.0s (acceleration phase)
            val t = i * 0.100
            val speed = source.calculateSpeedMs(t)

            assertTrue("Speed must increase monotonically during acceleration phase at t=$t", speed >= prevSpeed)
            val delta = speed - prevSpeed
            assertTrue("Frame-to-frame change ($delta m/s) must be smooth and bounded", delta <= maxAllowedDeltaPer100ms)
            prevSpeed = speed
        }

        // At end of acceleration (t=8.0s), speed should be close to base cruise speed (~41 km/h = 11.39 m/s)
        val speedAt8sKmh = source.calculateSpeedKmh(8.0)
        assertEquals(41.0, speedAt8sKmh, 0.1)
    }

    @Test
    fun test4_CruiseStability() {
        val source = DemoMotionSource()

        // 8.0s to 30.0s (cruise phase)
        for (i in 80..300) {
            val t = i * 0.100
            val speedKmh = source.calculateSpeedKmh(t)
            assertTrue("Cruise speed ($speedKmh km/h) at t=$t must stay within [40.0, 42.0] km/h", speedKmh in 40.0..42.0)
        }
    }

    @Test
    fun test5_NoInvalidValues() {
        val source = DemoMotionSource()

        val testTimes = doubleArrayOf(
            -1.0, 0.0, 0.5, 2.0, 5.0, 8.0, 10.0, 50.0, 100.0,
            Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY
        )

        for (t in testTimes) {
            val speedMs = source.calculateSpeedMs(t)
            val speedKmh = source.calculateSpeedKmh(t)

            assertFalse("Speed must not be NaN for t=$t", speedMs.isNaN())
            assertFalse("Speed must not be Infinite for t=$t", speedMs.isInfinite())
            assertTrue("Speed must be non-negative for t=$t", speedMs >= 0.0)

            assertFalse("Speed km/h must not be NaN for t=$t", speedKmh.isNaN())
            assertFalse("Speed km/h must not be Infinite for t=$t", speedKmh.isInfinite())
            assertTrue("Speed km/h must be non-negative for t=$t", speedKmh >= 0.0)
        }
    }

    @Test
    fun test6_BoundedAccelerationDeceleration() {
        val source = DemoMotionSource()
        val dt = 0.100

        var maxAccel = 0.0
        var maxDecel = 0.0

        var prevSpeed = source.calculateSpeedMs(0.0)
        for (i in 1..300) { // 0.1s to 30.0s
            val t = i * dt
            val currentSpeed = source.calculateSpeedMs(t)
            val accel = (currentSpeed - prevSpeed) / dt

            if (accel > 0) {
                maxAccel = maxOf(maxAccel, accel)
            } else {
                maxDecel = maxOf(maxDecel, abs(accel))
            }
            prevSpeed = currentSpeed
        }

        // Realistic passenger vehicle acceleration bounds:
        // Max acceleration <= 3.5 m/s^2 (~0.36g)
        // Max deceleration <= 1.0 m/s^2 during gentle breathing cruise
        assertTrue("Max acceleration ($maxAccel m/s^2) must be <= 3.5 m/s^2", maxAccel <= 3.5)
        assertTrue("Max deceleration ($maxDecel m/s^2) must be <= 1.0 m/s^2", maxDecel <= 1.0)
    }

    @Test
    fun test7_DeterministicCorridorHeading() {
        val source1 = DemoMotionSource()
        val source2 = DemoMotionSource()

        val steps = 300 // 30.0s @ 10Hz
        for (i in 0..steps) {
            val t = i * 0.100
            val h1 = source1.calculateHeadingDeg(t)
            val h2 = source2.calculateHeadingDeg(t)
            assertEquals("Heading at t=$t must be strictly deterministic", h1, h2, 0.0)
            assertTrue("Heading at t=$t must be within [0, 360)", h1 in 0.0..<360.0)
        }
    }

    @Test
    fun test8_DisplacementMatchesSpeedAndDt() {
        val source = DemoMotionSource()
        val dt = 0.100

        for (i in 0..300) {
            val t = i * dt
            val speed = source.calculateSpeedMs(t)
            val heading = source.calculateHeadingDeg(t)
            val (dEast, dNorth) = source.calculateDisplacement(speed, heading, dt)

            val displacementMagnitude = kotlin.math.hypot(dEast, dNorth)
            val expectedDisplacement = speed * dt
            assertEquals("Displacement magnitude at t=$t must equal speed * dt", expectedDisplacement, displacementMagnitude, 1e-6)
        }
    }

    @Test
    fun test9_HeadingConsistency() {
        val source = DemoMotionSource()
        val dt = 0.100

        // Test cardinal directions and 45-degree corridor
        val (dEastNorth, dNorthNorth) = source.calculateDisplacement(10.0, 0.0, dt)
        assertEquals(0.0, dEastNorth, 1e-6)
        assertEquals(1.0, dNorthNorth, 1e-6)

        val (dEastEast, dNorthEast) = source.calculateDisplacement(10.0, 90.0, dt)
        assertEquals(1.0, dEastEast, 1e-6)
        assertEquals(0.0, dNorthEast, 1e-6)

        val (dEastNE, dNorthNE) = source.calculateDisplacement(10.0, 45.0, dt)
        assertEquals(1.0 / kotlin.math.sqrt(2.0), dEastNE, 1e-6)
        assertEquals(1.0 / kotlin.math.sqrt(2.0), dNorthNE, 1e-6)
    }

    @Test
    fun test10_SmoothContinuousCurvature() {
        val source = DemoMotionSource()
        val dt = 0.100

        var prevHeading = source.calculateHeadingDeg(0.0)
        for (i in 1..300) {
            val t = i * dt
            val h = source.calculateHeadingDeg(t)
            val headingRate = abs(h - prevHeading) / dt
            // Max turning rate should be bounded and gentle (< 10 deg/s)
            assertTrue("Heading rate ($headingRate deg/s) at t=$t must remain smooth and gentle (< 10 deg/s)", headingRate < 10.0)
            prevHeading = h
        }
    }
}

