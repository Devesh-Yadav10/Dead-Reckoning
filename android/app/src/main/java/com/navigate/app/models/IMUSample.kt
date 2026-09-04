package com.navigate.app.models

/**
 * Standardized 6-axis IMU sample.
 *
 * Coordinate Frame: Vehicle Forward-Left-Up (FLU).
 * Units:
 *   - timestamp: seconds (monotonic or epoch timestamp)
 *   - ax, ay, az: m/s^2
 *   - gx, gy, gz: rad/s
 */
data class IMUSample(
    val timestamp: Double,
    val ax: Double,
    val ay: Double,
    val az: Double,
    val gx: Double,
    val gy: Double,
    val gz: Double
) {
    fun toDoubleArray(): DoubleArray = doubleArrayOf(ax, ay, az, gx, gy, gz)
}

