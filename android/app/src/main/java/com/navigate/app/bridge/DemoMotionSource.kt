package com.navigate.app.bridge

import kotlin.math.PI
import kotlin.math.sin

/**
 * DemoMotionSource — Deterministic vehicle motion generator for NAVIGATE 2.0 demonstrations.
 *
 * Implements a physically plausible, smooth vehicle speed profile:
 *   - Phase A (0.0s - 2.0s): Stationary rest (0.0 km/h)
 *   - Phase B (2.0s - 8.0s): Smooth S-curve (cubic Hermite) acceleration from 0 -> ~41 km/h
 *   - Phase C (8.0s+): Stable cruise at ~41.0 km/h with subtle, bounded periodic variation (40.2 - 41.8 km/h)
 */
class DemoMotionSource(
    val stationaryDurationS: Double = DEFAULT_STATIONARY_DURATION_S,
    val accelDurationS: Double = DEFAULT_ACCEL_DURATION_S,
    val cruiseBaseSpeedMs: Double = DEFAULT_CRUISE_BASE_SPEED_MS,
    val cruiseVariationMs: Double = DEFAULT_CRUISE_VARIATION_MS,
    val cruiseVariationPeriodS: Double = DEFAULT_CRUISE_VARIATION_PERIOD_S,
    val baseHeadingDeg: Double = DEFAULT_BASE_HEADING_DEG,
    val curveAmplitudeDeg: Double = DEFAULT_CURVE_AMPLITUDE_DEG,
    val curvePeriodS: Double = DEFAULT_CURVE_PERIOD_S
) {
    companion object {
        const val DEFAULT_STATIONARY_DURATION_S = 2.0
        const val DEFAULT_ACCEL_DURATION_S = 6.0
        const val DEFAULT_CRUISE_BASE_SPEED_MS = 41.0 / 3.6     // ~11.3889 m/s (41.0 km/h)
        const val DEFAULT_CRUISE_VARIATION_MS = 0.8 / 3.6       // ~0.2222 m/s (0.8 km/h variation)
        const val DEFAULT_CRUISE_VARIATION_PERIOD_S = 4.0      // 4-second breathing cycle
        const val DEFAULT_BASE_HEADING_DEG = 45.0               // Northeast corridor (Kingsway)
        const val DEFAULT_CURVE_AMPLITUDE_DEG = 12.0            // +/- 12° gentle corridor curve
        const val DEFAULT_CURVE_PERIOD_S = 20.0                 // 20s gentle curve cycle
    }

    val accelEndTimeS: Double = stationaryDurationS + accelDurationS

    /**
     * Calculates the deterministic vehicle speed in m/s at elapsed time [timeS].
     *
     * @param timeS Elapsed simulation time in seconds
     * @return Vehicle forward speed in m/s (non-negative, finite)
     */
    fun calculateSpeedMs(timeS: Double): Double {
        if (timeS.isNaN() || timeS.isInfinite() || timeS <= stationaryDurationS) {
            return 0.0
        }

        if (timeS < accelEndTimeS) {
            // Phase B: Smoothstep S-curve (3*tau^2 - 2*tau^3)
            val tau = (timeS - stationaryDurationS) / accelDurationS
            val smoothstep = 3.0 * tau * tau - 2.0 * tau * tau * tau
            return cruiseBaseSpeedMs * smoothstep
        }

        // Phase C: Stable cruise with bounded periodic variation (40.2 - 41.8 km/h)
        val cruiseElapsed = timeS - accelEndTimeS
        val variation = cruiseVariationMs * sin(2.0 * PI * cruiseElapsed / cruiseVariationPeriodS)
        return maxOf(0.0, cruiseBaseSpeedMs + variation)
    }

    /**
     * Calculates the deterministic vehicle speed in km/h at elapsed time [timeS].
     */
    fun calculateSpeedKmh(timeS: Double): Double {
        return calculateSpeedMs(timeS) * 3.6
    }

    /**
     * Calculates the deterministic vehicle heading in degrees at elapsed time [timeS].
     *
     * @param timeS Elapsed simulation time in seconds
     * @param initialHeading Initial reference heading in degrees (default: baseHeadingDeg)
     * @return Deterministic vehicle heading in degrees in [0, 360)
     */
    fun calculateHeadingDeg(
        timeS: Double,
        initialHeading: Double = baseHeadingDeg
    ): Double {
        if (timeS.isNaN() || timeS.isInfinite() || timeS < 0.0) {
            return normalizeHeading(initialHeading)
        }
        val curveOffset = curveAmplitudeDeg * sin(2.0 * PI * timeS / curvePeriodS)
        return normalizeHeading(initialHeading + curveOffset)
    }

    /**
     * Computes the local ENU displacement (dEast, dNorth) in meters for a given timestep [dt]
     * given current speed [speedMs] and heading [headingDeg].
     */
    fun calculateDisplacement(speedMs: Double, headingDeg: Double, dt: Double): Pair<Double, Double> {
        if (speedMs <= 0.0 || dt <= 0.0 || speedMs.isNaN() || dt.isNaN()) {
            return Pair(0.0, 0.0)
        }
        val headingRad = Math.toRadians(headingDeg)
        val dEast = speedMs * sin(headingRad) * dt
        val dNorth = speedMs * kotlin.math.cos(headingRad) * dt
        return Pair(dEast, dNorth)
    }

    private fun normalizeHeading(deg: Double): Double {
        var h = deg % 360.0
        if (h < 0.0) h += 360.0
        return h
    }
}

