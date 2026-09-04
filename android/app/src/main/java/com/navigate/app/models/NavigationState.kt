package com.navigate.app.models

/**
 * Instantaneous navigation state solution emitted by NAVIGATE 2.0.
 */
data class NavigationState(
    val timestamp: Double,
    val latitude: Double,
    val longitude: Double,
    val posEastM: Double,
    val posNorthM: Double,
    val posUpM: Double = 0.0,
    val speedMs: Double,
    val headingDeg: Double,
    val gnssState: GNSSState = GNSSState.AVAILABLE,
    val blackoutActive: Boolean = false,
    val roadConstraintActive: Boolean = false,
    val matchedWayId: Long? = null,
    val matchedRoadName: String? = null,
    val latencyMs: Double = 0.0,
    val diagnostics: Map<String, Any> = emptyMap()
) {
    /** Speed converted to km/h for UI display */
    val speedKmh: Double
        get() = speedMs * 3.6

    /** Formatted heading string, e.g. "045°" */
    val formattedHeading: String
        get() = String.format("%03.0f°", ((headingDeg % 360.0) + 360.0) % 360.0)

    /** Formatted speed string, e.g. "45.2 km/h" */
    val formattedSpeed: String
        get() = String.format("%.1f km/h", speedKmh)
}
