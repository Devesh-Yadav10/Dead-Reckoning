package com.navigate.app.models

/**
 * Immutable destination model representing a geocoded target destination.
 */
data class Destination(
    val latitude: Double,
    val longitude: Double,
    val displayName: String,
    val address: String = ""
)

