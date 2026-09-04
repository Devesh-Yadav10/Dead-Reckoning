package com.navigate.app.bridge

import com.navigate.app.models.Route
import org.osmdroid.util.GeoPoint
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * RouteTrajectoryGenerator — Deterministic distance-based route interpolation engine.
 *
 * Interpolates vehicle position and heading smoothly along OSRM route geometry in a local ENU metric frame.
 */
class RouteTrajectoryGenerator(
    route: Route,
    initialLat: Double = route.geometry.firstOrNull()?.latitude ?: 51.5074,
    initialLon: Double = route.geometry.firstOrNull()?.longitude ?: -0.1278,
    initialHeadingDeg: Double = 0.0
) {
    data class TrajectoryPoint(
        val latitude: Double,
        val longitude: Double,
        val eastM: Double,
        val northM: Double,
        val headingDeg: Double,
        val progress: Double,
        val isCompleted: Boolean
    )

    private val anchorLat: Double
    private val anchorLon: Double
    private val eastPoints: DoubleArray
    private val northPoints: DoubleArray
    private val cumDistances: DoubleArray
    private val segLengths: DoubleArray

    val totalDistanceMeters: Double
    val isValid: Boolean

    var travelledDistanceMeters: Double = 0.0
        private set

    var currentHeadingDeg: Double = initialHeadingDeg
        private set

    var isCompleted: Boolean = false
        private set

    init {
        val geo = route.geometry
        if (geo.size < 2 || geo.any { it.latitude.isNaN() || it.longitude.isNaN() || it.latitude.isInfinite() || it.longitude.isInfinite() }) {
            anchorLat = initialLat
            anchorLon = initialLon
            eastPoints = DoubleArray(0)
            northPoints = DoubleArray(0)
            cumDistances = DoubleArray(0)
            segLengths = DoubleArray(0)
            totalDistanceMeters = 0.0
            isValid = false
        } else {
            anchorLat = geo.first().latitude
            anchorLon = geo.first().longitude

            val n = geo.size
            eastPoints = DoubleArray(n)
            northPoints = DoubleArray(n)
            cumDistances = DoubleArray(n)
            segLengths = DoubleArray(max(0, n - 1))

            for (i in 0 until n) {
                val dLat = Math.toRadians(geo[i].latitude - anchorLat)
                val dLon = Math.toRadians(geo[i].longitude - anchorLon)
                northPoints[i] = dLat * 6371000.0
                eastPoints[i] = dLon * (6371000.0 * cos(Math.toRadians(anchorLat)))
            }

            cumDistances[0] = 0.0
            for (i in 0 until n - 1) {
                val dx = eastPoints[i + 1] - eastPoints[i]
                val dy = northPoints[i + 1] - northPoints[i]
                val len = hypot(dx, dy)
                segLengths[i] = len
                cumDistances[i + 1] = cumDistances[i] + len
            }

            totalDistanceMeters = cumDistances[n - 1]
            isValid = totalDistanceMeters > 0.01

            if (isValid) {
                // Initialize progress nearest to initial position
                val curDLat = Math.toRadians(initialLat - anchorLat)
                val curDLon = Math.toRadians(initialLon - anchorLon)
                val curNorth = curDLat * 6371000.0
                val curEast = curDLon * (6371000.0 * cos(Math.toRadians(anchorLat)))

                travelledDistanceMeters = findNearestDistanceAlongRoute(curEast, curNorth)
                // Initialize heading to match the active segment
                val initialPt = sampleAtDistance(travelledDistanceMeters)
                currentHeadingDeg = initialPt.headingDeg
            }
        }
    }

    /**
     * Advances the trajectory by [distanceDeltaMeters] and returns the updated [TrajectoryPoint].
     */
    fun advance(
        distanceDeltaMeters: Double,
        currentSpeedMs: Double = 11.4,
        dt: Double = 0.100,
        maxTurnRateDegPerSec: Double = 90.0
    ): TrajectoryPoint {
        if (!isValid) {
            return TrajectoryPoint(anchorLat, anchorLon, 0.0, 0.0, currentHeadingDeg, 0.0, isCompleted = true)
        }

        travelledDistanceMeters = min(totalDistanceMeters, max(0.0, travelledDistanceMeters + distanceDeltaMeters))
        val sampled = sampleAtDistance(travelledDistanceMeters)

        // Smooth heading towards segment heading
        val targetHeading = sampled.headingDeg
        val diff = ((targetHeading - currentHeadingDeg + 540.0) % 360.0) - 180.0
        val maxTurn = maxTurnRateDegPerSec * dt
        val turnStep = min(maxTurn, max(-maxTurn, diff))
        currentHeadingDeg = (currentHeadingDeg + turnStep + 360.0) % 360.0

        isCompleted = travelledDistanceMeters >= totalDistanceMeters - 1e-4

        return TrajectoryPoint(
            latitude = sampled.latitude,
            longitude = sampled.longitude,
            eastM = sampled.eastM,
            northM = sampled.northM,
            headingDeg = currentHeadingDeg,
            progress = if (totalDistanceMeters > 0.0) min(1.0, travelledDistanceMeters / totalDistanceMeters) else 1.0,
            isCompleted = isCompleted
        )
    }

    /**
     * Interpolates position and segment heading at [distMeters].
     */
    fun sampleAtDistance(distMeters: Double): TrajectoryPoint {
        if (!isValid) {
            return TrajectoryPoint(anchorLat, anchorLon, 0.0, 0.0, currentHeadingDeg, 0.0, isCompleted = true)
        }

        val clampedDist = min(totalDistanceMeters, max(0.0, distMeters))
        val n = eastPoints.size

        // Locate segment index
        var segIdx = 0
        while (segIdx < n - 2 && cumDistances[segIdx + 1] < clampedDist) {
            segIdx++
        }

        val segStartDist = cumDistances[segIdx]
        val segLen = segLengths[segIdx]
        val u = if (segLen > 1e-6) min(1.0, max(0.0, (clampedDist - segStartDist) / segLen)) else 0.0

        val east = eastPoints[segIdx] + u * (eastPoints[segIdx + 1] - eastPoints[segIdx])
        val north = northPoints[segIdx] + u * (northPoints[segIdx + 1] - northPoints[segIdx])

        // Convert ENU -> WGS84
        val lat = anchorLat + (north / 6371000.0) * (180.0 / PI)
        val lon = anchorLon + (east / (6371000.0 * cos(Math.toRadians(anchorLat)))) * (180.0 / PI)

        // Raw segment heading in degrees [0, 360)
        val dx = eastPoints[segIdx + 1] - eastPoints[segIdx]
        val dy = northPoints[segIdx + 1] - northPoints[segIdx]
        val headingRad = atan2(dx, dy)
        var headingDeg = Math.toDegrees(headingRad) % 360.0
        if (headingDeg < 0.0) headingDeg += 360.0

        val progress = if (totalDistanceMeters > 0.0) min(1.0, clampedDist / totalDistanceMeters) else 1.0
        val done = clampedDist >= totalDistanceMeters - 1e-4

        return TrajectoryPoint(lat, lon, east, north, headingDeg, progress, done)
    }

    /**
     * Finds the distance along the route closest to the point ([eastM], [northM]).
     */
    fun findNearestDistanceAlongRoute(eastM: Double, northM: Double): Double {
        if (!isValid) return 0.0
        var bestDistAlongRoute = 0.0
        var minSqDist = Double.MAX_VALUE

        for (i in 0 until eastPoints.size - 1) {
            val x1 = eastPoints[i]
            val y1 = northPoints[i]
            val x2 = eastPoints[i + 1]
            val y2 = northPoints[i + 1]

            val dx = x2 - x1
            val dy = y2 - y1
            val segLenSq = dx * dx + dy * dy

            val u = if (segLenSq > 1e-6) {
                min(1.0, max(0.0, ((eastM - x1) * dx + (northM - y1) * dy) / segLenSq))
            } else {
                0.0
            }

            val projX = x1 + u * dx
            val projY = y1 + u * dy
            val sqDistToProj = (eastM - projX) * (eastM - projX) + (northM - projY) * (northM - projY)

            if (sqDistToProj < minSqDist) {
                minSqDist = sqDistToProj
                bestDistAlongRoute = cumDistances[i] + u * segLengths[i]
            }
        }
        return bestDistAlongRoute
    }
}

