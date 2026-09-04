package com.navigate.app.models

/**
 * Standardized GNSS position fix.
 *
 * Units:
 *   - timestamp: seconds
 *   - latitude, longitude: degrees in WGS84
 *   - altitude: metres above WGS84 ellipsoid
 *   - accuracyM: 1-sigma horizontal accuracy in metres
 */
data class GNSSSample(
    val timestamp: Double,
    val latitude: Double,
    val longitude: Double,
    val altitude: Double = 0.0,
    val accuracyM: Double = 2.5,
    val speedMs: Double? = null,
    val bearingDeg: Double? = null
)

