package com.navigate.app

import com.navigate.app.models.IMUSample
import com.navigate.app.sensors.AndroidSensorAdapter
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class SensorAdapterTest {

    @Test
    fun testCoordinateTransformationPortrait() {
        val timestamp = 100.500
        val rawAx = 1.2   // Phone +X (Right)
        val rawAy = 3.4   // Phone +Y (Up/Forward)
        val rawAz = 9.81  // Phone +Z (Out of screen)
        val rawGx = 0.05
        val rawGy = 0.10
        val rawGz = 0.15

        val sample: IMUSample = AndroidSensorAdapter.convertToVehicleBodyFrame(
            timestampSec = timestamp,
            rawAx = rawAx,
            rawAy = rawAy,
            rawAz = rawAz,
            rawGx = rawGx,
            rawGy = rawGy,
            rawGz = rawGz
        )

        assertNotNull(sample)
        assertEquals(timestamp, sample.timestamp, 1e-6)
        // Vehicle Forward (+X_v) = Phone +Y
        assertEquals(3.4, sample.ax, 1e-6)
        // Vehicle Left (+Y_v) = Phone -X
        assertEquals(-1.2, sample.ay, 1e-6)
        // Vehicle Up (+Z_v) = Phone +Z
        assertEquals(9.81, sample.az, 1e-6)

        assertEquals(0.10, sample.gx, 1e-6)
        assertEquals(-0.05, sample.gy, 1e-6)
        assertEquals(0.15, sample.gz, 1e-6)
    }

    @Test
    fun testRawIMUSamplePreservesAsymmetricChannels() {
        val timestamp = 100.500
        val rawAx = 1.11  // Phone +X
        val rawAy = 2.22  // Phone +Y
        val rawAz = 3.33  // Phone +Z
        val rawGx = 4.44  // Gyro X
        val rawGy = 5.55  // Gyro Y
        val rawGz = 6.66  // Gyro Z

        val sample = AndroidSensorAdapter.createRawIMUSample(
            timestampSec = timestamp,
            rawAx = rawAx,
            rawAy = rawAy,
            rawAz = rawAz,
            rawGx = rawGx,
            rawGy = rawGy,
            rawGz = rawGz
        )

        assertNotNull(sample)
        assertEquals(timestamp, sample.timestamp, 1e-6)
        // Verify exact asymmetric channels (must NOT be permuted to [ay, -ax, az, gy, -gx, gz])
        assertEquals(1.11, sample.ax, 1e-6)
        assertEquals(2.22, sample.ay, 1e-6)
        assertEquals(3.33, sample.az, 1e-6)
        assertEquals(4.44, sample.gx, 1e-6)
        assertEquals(5.55, sample.gy, 1e-6)
        assertEquals(6.66, sample.gz, 1e-6)
    }

    @Test
    fun testIMUSampleToDoubleArray() {
        val sample = IMUSample(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        val arr = sample.toDoubleArray()
        assertEquals(6, arr.size)
        assertEquals(2.0, arr[0], 1e-6)
        assertEquals(7.0, arr[5], 1e-6)
    }
}

