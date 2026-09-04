package com.navigate.app.services

import com.navigate.app.models.Route
import com.navigate.app.models.RouteRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.osmdroid.util.GeoPoint
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale

/**
 * OsrmRouteService — OpenStreetMap OSRM routing service implementation.
 */
class OsrmRouteService(
    private val baseUrl: String = "https://router.project-osrm.org/route/v1/driving",
    private val userAgent: String = "NAVIGATE-2.0-Android/2.0 (hackathon-demo)"
) : RouteService {

    override suspend fun getRoute(request: RouteRequest): Result<Route> = withContext(Dispatchers.IO) {
        var connection: HttpURLConnection? = null
        try {
            // OSRM URL format: {lon1},{lat1};{lon2},{lat2}
            val urlString = String.format(
                Locale.US,
                "%s/%.6f,%.6f;%.6f,%.6f?overview=full&geometries=geojson",
                baseUrl,
                request.originLongitude,
                request.originLatitude,
                request.destinationLongitude,
                request.destinationLatitude
            )
            val url = URL(urlString)
            connection = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                setRequestProperty("User-Agent", userAgent)
                setRequestProperty("Accept", "application/json")
                connectTimeout = 5000
                readTimeout = 5000
            }

            val responseCode = connection.responseCode
            if (responseCode != HttpURLConnection.HTTP_OK) {
                return@withContext Result.failure(Exception("HTTP $responseCode: ${connection.responseMessage}"))
            }

            val responseText = connection.inputStream.bufferedReader().use { it.readText() }
            val route = parseResponse(responseText)
            Result.success(route)
        } catch (e: Exception) {
            Result.failure(e)
        } finally {
            connection?.disconnect()
        }
    }

    /**
     * Parses OSRM GeoJSON routing response into a [Route] object.
     */
    fun parseResponse(jsonString: String): Route {
        val root = JSONObject(jsonString)
        val code = root.optString("code", "")
        if (code != "Ok") {
            throw IllegalArgumentException("OSRM routing returned code: $code")
        }

        val routes = root.getJSONArray("routes")
        if (routes.length() == 0) {
            throw IllegalArgumentException("No route found in response")
        }

        val firstRoute = routes.getJSONObject(0)
        val distance = firstRoute.optDouble("distance", 0.0)
        val duration = firstRoute.optDouble("duration", 0.0)

        val geometryObj = firstRoute.getJSONObject("geometry")
        val coordinates = geometryObj.getJSONArray("coordinates")
        if (coordinates.length() == 0) {
            throw IllegalArgumentException("Route geometry is empty")
        }

        val points = mutableListOf<GeoPoint>()
        for (i in 0 until coordinates.length()) {
            val pointArray = coordinates.getJSONArray(i)
            // GeoJSON coordinates are in [longitude, latitude] order
            val lon = pointArray.getDouble(0)
            val lat = pointArray.getDouble(1)
            points.add(GeoPoint(lat, lon))
        }

        return Route(
            geometry = points,
            distanceMeters = distance,
            durationSeconds = duration
        )
    }
}

