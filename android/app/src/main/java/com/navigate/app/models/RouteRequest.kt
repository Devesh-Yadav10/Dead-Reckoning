package com.navigate.app.models

/**
 * RouteRequest — Origin and destination coordinates for routing.
 */
data class RouteRequest(
    val originLatitude: Double,
    val originLongitude: Double,
    val destinationLatitude: Double,
    val destinationLongitude: Double
)

