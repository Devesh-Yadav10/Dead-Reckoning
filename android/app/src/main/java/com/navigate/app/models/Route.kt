package com.navigate.app.models

import org.osmdroid.util.GeoPoint
import kotlin.math.roundToLong

/**
 * Route — Calculated route solution containing polyline geometry, distance, and duration.
 */
data class Route(
    val geometry: List<GeoPoint>,
    val distanceMeters: Double,
    val durationSeconds: Double
) {
    val formattedDistance: String
        get() = if (distanceMeters >= 1000.0) {
            String.format("%.1f km", distanceMeters / 1000.0)
        } else {
            String.format("%.0f m", distanceMeters)
        }

    val formattedDuration: String
        get() {
            val minutes = (durationSeconds / 60.0).roundToLong()
            return if (minutes >= 60) {
                val hours = minutes / 60
                val remMin = minutes % 60
                "${hours}h ${remMin}m"
            } else {
                "${maxOf(1L, minutes)} min"
            }
        }

    val summary: String
        get() = "$formattedDistance · $formattedDuration"
}

