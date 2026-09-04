package com.navigate.app.services

import com.navigate.app.models.Destination
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * NominatimDestinationSearchService — OpenStreetMap Nominatim geocoding provider.
 */
class NominatimDestinationSearchService(
    private val baseUrl: String = "https://nominatim.openstreetmap.org/search",
    private val userAgent: String = "NAVIGATE-2.0-Android/2.0 (hackathon-demo)"
) : DestinationSearchService {

    override suspend fun search(query: String): Result<List<Destination>> = withContext(Dispatchers.IO) {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) {
            return@withContext Result.success(emptyList())
        }

        var connection: HttpURLConnection? = null
        try {
            val encodedQuery = URLEncoder.encode(trimmed, "UTF-8")
            val urlString = "$baseUrl?q=$encodedQuery&format=json&addressdetails=1&limit=5"
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
            val destinations = parseResponse(responseText)
            Result.success(destinations)
        } catch (e: Exception) {
            Result.failure(e)
        } finally {
            connection?.disconnect()
        }
    }

    /**
     * Parses Nominatim JSON response into a list of [Destination] objects.
     */
    fun parseResponse(jsonString: String): List<Destination> {
        val results = mutableListOf<Destination>()
        val array = JSONArray(jsonString)
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            val lat = obj.getDouble("lat")
            val lon = obj.getDouble("lon")
            val displayName = obj.optString("display_name", "")

            // Parse concise title and address details
            val parts = displayName.split(",").map { it.trim() }
            val primaryName = parts.firstOrNull() ?: displayName
            val secondaryAddress = if (parts.size > 1) parts.drop(1).take(3).joinToString(", ") else ""

            results.add(
                Destination(
                    latitude = lat,
                    longitude = lon,
                    displayName = primaryName,
                    address = secondaryAddress.ifEmpty { displayName }
                )
            )
        }
        return results
    }
}

