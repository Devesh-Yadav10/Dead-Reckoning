package com.navigate.app.sensors

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.Looper
import android.util.Log
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationAvailability
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.navigate.app.models.GNSSSample

/**
 * AndroidLocationAdapter — Streams GNSS position fixes from Google Play Services FusedLocationProviderClient.
 *
 * Implements high-accuracy 1 Hz GNSS streaming with:
 *   - FusedLocationProvider (GPS + Wi-Fi + Cell + Sensors)
 *   - Explicit permission & provider failure handling
 *   - Initial last-known location fallback (clearly distinguished from fresh fixes)
 *   - Stale last-known location rejection once fresh fix is active
 *   - Structured diagnostic logging
 */
class AndroidLocationAdapter(
    private val context: Context,
    private val fusedLocationClient: FusedLocationProviderClient? = null,
    private val onError: ((Throwable) -> Unit)? = null,
    private val onGNSSFixReady: (GNSSSample) -> Unit
) {

    constructor(
        context: Context,
        onGNSSFixReady: (GNSSSample) -> Unit
    ) : this(context, null, null, onGNSSFixReady)

    private val client: FusedLocationProviderClient by lazy {
        fusedLocationClient ?: LocationServices.getFusedLocationProviderClient(context)
    }

    var isListening = false
        private set

    var hasReceivedFreshFix = false
        private set

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            for (location in result.locations) {
                processLocation(location, isLastKnown = false)
            }
        }

        override fun onLocationAvailability(availability: LocationAvailability) {
            Log.d("NAVIGATE_LOCATION", "[LocationAvailability] isLocationAvailable=${availability.isLocationAvailable}")
        }
    }

    @SuppressLint("MissingPermission")
    fun startListening() {
        if (isListening) {
            Log.d("NAVIGATE_LOCATION", "[FusedLocationProvider] startListening() called while already listening")
            return
        }

        try {
            // 1. Initial fallback: check last-known location if fresh fix hasn't arrived
            client.lastLocation
                .addOnSuccessListener { loc ->
                    if (loc != null && !hasReceivedFreshFix) {
                        Log.i("NAVIGATE_LOCATION", "[FusedLocationProvider] Found last-known location: lat=${loc.latitude}, lon=${loc.longitude}, accuracy=${if (loc.hasAccuracy()) "%.2fm".format(loc.accuracy) else "N/A"}")
                        processLocation(loc, isLastKnown = true)
                    }
                }
                .addOnFailureListener { e ->
                    Log.w("NAVIGATE_LOCATION", "[FusedLocationProvider] Failed to fetch last-known location: ${e.message}")
                }

            // 2. Continuous 1 Hz High-Accuracy Live Updates
            val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
                .setMinUpdateIntervalMillis(500L)
                .setMinUpdateDistanceMeters(0f)
                .setWaitForAccurateLocation(false)
                .build()

            val looper = Looper.myLooper() ?: Looper.getMainLooper()
            client.requestLocationUpdates(locationRequest, locationCallback, looper)
                .addOnSuccessListener {
                    isListening = true
                    Log.i("NAVIGATE_LOCATION", "[FusedLocationProvider] Successfully registered 1 Hz HIGH_ACCURACY location updates")
                }
                .addOnFailureListener { e ->
                    isListening = false
                    Log.e("NAVIGATE_LOCATION", "[FusedLocationProvider] Failed to request location updates: ${e.message}", e)
                    onError?.invoke(e)
                }
        } catch (e: SecurityException) {
            isListening = false
            Log.e("NAVIGATE_LOCATION", "[FusedLocationProvider] Missing location permissions: ${e.message}", e)
            onError?.invoke(e)
        } catch (e: Exception) {
            isListening = false
            Log.e("NAVIGATE_LOCATION", "[FusedLocationProvider] Error starting location updates: ${e.message}", e)
            onError?.invoke(e)
        }
    }

    fun stopListening() {
        if (!isListening) return
        try {
            client.removeLocationUpdates(locationCallback)
            Log.i("NAVIGATE_LOCATION", "[FusedLocationProvider] Successfully unregistered location updates")
        } catch (e: Exception) {
            Log.w("NAVIGATE_LOCATION", "[FusedLocationProvider] Error removing location updates: ${e.message}")
        }
        isListening = false
    }

    fun onLocationChanged(location: Location) {
        processLocation(location, isLastKnown = false)
    }

    fun processLocation(location: Location, isLastKnown: Boolean = false) {
        if (isLastKnown && hasReceivedFreshFix) {
            Log.d("NAVIGATE_LOCATION", "[FusedLocationProvider] Discarding stale last-known fix because fresh fix is already active")
            return
        }

        if (!isLastKnown) {
            hasReceivedFreshFix = true
        }

        val rawSpeed = if (location.hasSpeed()) location.speed.toDouble() else null
        val sourceLabel = if (isLastKnown) "LAST_KNOWN" else "LIVE_FUSED"
        val providerStr = location.provider ?: "fused"

        Log.i(
            "NAVIGATE_LOCATION",
            "[GNSS Fix] source=$sourceLabel($providerStr), " +
                    "lat=${"%.6f".format(location.latitude)}, lon=${"%.6f".format(location.longitude)}, " +
                    "accuracy=${if (location.hasAccuracy()) "%.2fm".format(location.accuracy) else "N/A"}, " +
                    "speed=${rawSpeed?.let { "%.2f m/s (%.1f km/h)".format(it, it * 3.6) } ?: "null"}, " +
                    "bearing=${if (location.hasBearing()) "%.1f°".format(location.bearing) else "null"}, " +
                    "time=${location.time}"
        )

        val sample = GNSSSample(
            timestamp = location.time / 1000.0,
            latitude = location.latitude,
            longitude = location.longitude,
            altitude = if (location.hasAltitude()) location.altitude else 0.0,
            accuracyM = if (location.hasAccuracy()) location.accuracy.toDouble() else 2.5,
            speedMs = rawSpeed,
            bearingDeg = if (location.hasBearing()) location.bearing.toDouble() else null
        )
        onGNSSFixReady(sample)
    }

    fun reset() {
        stopListening()
        hasReceivedFreshFix = false
    }
}

