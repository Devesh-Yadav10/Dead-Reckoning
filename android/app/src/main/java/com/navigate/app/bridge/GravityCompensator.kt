package com.navigate.app.bridge

import com.navigate.app.models.IMUSample
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * GravityCompensator — Real-time gravity estimation, orientation leveling, and linear acceleration extraction.
 *
 * Prevents phone tilt/rotation from inducing false forward acceleration into AI velocity inference:
 *   1. Tracks 3D gravity vector in the device body frame via gyro propagation + adaptive accelerometer complementary filtering.
 *   2. Computes true device-frame linear acceleration: a_lin = a_measured - g_body.
 *   3. Constructs a leveling rotation matrix R_level that rotates the estimated gravity vector to nominal vertical [0, 0, 9.80665].
 *   4. Transforms linear acceleration into the model's nominal reference frame: a_comp = R_level * a_lin + [0, 0, 9.80665].
 *   5. Compensates full 50-sample IMU windows for ONNX model inference.
 */
class GravityCompensator(
    private val standardGravity: Double = 9.80665,
    private val alphaAccel: Double = 0.08
) {
    data class CompensationResult(
        val gravityX: Double,
        val gravityY: Double,
        val gravityZ: Double,
        val linearAx: Double,
        val linearAy: Double,
        val linearAz: Double,
        val linearAccelMag: Double,
        val compAx: Double,
        val compAy: Double,
        val compAz: Double,
        val compGx: Double,
        val compGy: Double,
        val compGz: Double
    )

    private var gx = 0.0
    private var gy = 0.0
    private var gz = standardGravity
    private var isInitialized = false

    val currentGravityX: Double get() = gx
    val currentGravityY: Double get() = gy
    val currentGravityZ: Double get() = gz

    fun reset() {
        gx = 0.0
        gy = 0.0
        gz = standardGravity
        isInitialized = false
    }

    /**
     * Ingests a 10 Hz IMUSample, updates the continuous gravity tracker, and returns compensation results.
     */
    fun process(sample: IMUSample, dt: Double = 0.100): CompensationResult {
        val ax = sample.ax
        val ay = sample.ay
        val az = sample.az
        val omx = sample.gx
        val omy = sample.gy
        val omz = sample.gz

        val accelMag = sqrt(ax * ax + ay * ay + az * az)

        if (!isInitialized) {
            if (accelMag > 1e-3) {
                gx = (ax / accelMag) * standardGravity
                gy = (ay / accelMag) * standardGravity
                gz = (az / accelMag) * standardGravity
                isInitialized = true
            } else {
                gx = 0.0
                gy = 0.0
                gz = standardGravity
                isInitialized = true
            }
        } else {
            // 1. Gyroscope propagation: dG/dt = G x Omega
            val omMag = sqrt(omx * omx + omy * omy + omz * omz)
            if (omMag > 1e-4) {
                val dgX = (gy * omz - gz * omy) * dt
                val dgY = (gz * omx - gx * omz) * dt
                val dgZ = (gx * omy - gy * omx) * dt

                gx += dgX
                gy += dgY
                gz += dgZ
            }

            // 2. Accelerometer complementary correction:
            // Only correct gravity when the device is quasistatic (near 1g and low angular rate).
            // This prevents translational vehicle acceleration from dragging the gravity vector.
            val diffFrom1g = abs(accelMag - standardGravity)
            if (diffFrom1g < 0.25 && omMag < 0.05 && accelMag > 1e-3) {
                val targetGx = (ax / accelMag) * standardGravity
                val targetGy = (ay / accelMag) * standardGravity
                val targetGz = (az / accelMag) * standardGravity

                gx += alphaAccel * (targetGx - gx)
                gy += alphaAccel * (targetGy - gy)
                gz += alphaAccel * (targetGz - gz)
            }

            // Normalize gravity vector to standard magnitude
            val gNorm = sqrt(gx * gx + gy * gy + gz * gz)
            if (gNorm > 1e-6) {
                gx = (gx / gNorm) * standardGravity
                gy = (gy / gNorm) * standardGravity
                gz = (gz / gNorm) * standardGravity
            }
        }

        // 3. Compute Linear Acceleration in Device Frame
        val linAx = ax - gx
        val linAy = ay - gy
        val linAz = az - gz
        val linAccelMag = sqrt(linAx * linAx + linAy * linAy + linAz * linAz)

        // 4. Construct Leveling Rotation Matrix
        // Aligns gravity vector [gx, gy, gz] with vertical [0, 0, standardGravity]
        val (rotLinAx, rotLinAy, rotLinAz) = rotateToLevelFrame(linAx, linAy, linAz, gx, gy, gz)
        val (rotGx, rotGy, rotGz) = rotateToLevelFrame(omx, omy, omz, gx, gy, gz)

        // 5. Compensated IMU channels in model reference frame
        // In the level frame, gravity is purely vertical (+Z = 9.80665)
        val compAx = rotLinAx
        val compAy = rotLinAy
        val compAz = rotLinAz + standardGravity

        return CompensationResult(
            gravityX = gx,
            gravityY = gy,
            gravityZ = gz,
            linearAx = linAx,
            linearAy = linAy,
            linearAz = linAz,
            linearAccelMag = linAccelMag,
            compAx = compAx,
            compAy = compAy,
            compAz = compAz,
            compGx = rotGx,
            compGy = rotGy,
            compGz = rotGz
        )
    }

    /**
     * Compensates a 50x6 window of raw [ax, ay, az, gx, gy, gz] samples using the tracked gravity vector.
     */
    fun compensateWindow(window: List<FloatArray>): List<FloatArray> {
        val gMag = sqrt(gx * gx + gy * gy + gz * gz)
        val currentGx = if (gMag > 1e-4) gx else 0.0
        val currentGy = if (gMag > 1e-4) gy else 0.0
        val currentGz = if (gMag > 1e-4) gz else standardGravity

        return window.map { raw ->
            val ax = raw[0].toDouble()
            val ay = raw[1].toDouble()
            val az = raw[2].toDouble()
            val gx = raw[3].toDouble()
            val gy = raw[4].toDouble()
            val gz = raw[5].toDouble()

            val linAx = ax - currentGx
            val linAy = ay - currentGy
            val linAz = az - currentGz

            val (rotAx, rotAy, rotAz) = rotateToLevelFrame(linAx, linAy, linAz, currentGx, currentGy, currentGz)
            val (rotGx, rotGy, rotGz) = rotateToLevelFrame(gx, gy, gz, currentGx, currentGy, currentGz)

            floatArrayOf(
                rotAx.toFloat(),
                rotAy.toFloat(),
                (rotAz + standardGravity).toFloat(),
                rotGx.toFloat(),
                rotGy.toFloat(),
                rotGz.toFloat()
            )
        }
    }

    /**
     * Rotates a vector (vx, vy, vz) from the tilted body frame to the gravity-leveled frame.
     */
    fun rotateToLevelFrame(
        vx: Double, vy: Double, vz: Double,
        gravX: Double, gravY: Double, gravZ: Double
    ): Triple<Double, Double, Double> {
        val gMag = sqrt(gravX * gravX + gravY * gravY + gravZ * gravZ)
        if (gMag < 1e-4) return Triple(vx, vy, vz)

        val nx = gravX / gMag
        val ny = gravY / gMag
        val nz = gravZ / gMag

        // Rotation axis v = n x [0, 0, 1] = [ny, -nx, 0]
        val vX = ny
        val vY = -nx

        val c = nz // cos(theta) = n . [0, 0, 1]
        val s = sqrt(vX * vX + vY * vY) // sin(theta)

        if (s < 1e-6) {
            // Already aligned or antiparallel
            return if (c > 0) {
                Triple(vx, vy, vz)
            } else {
                Triple(vx, -vy, -vz)
            }
        }

        // Rodrigues formula: v_rot = v*c + (k x v)*s + k*(k . v)*(1 - c)
        // Unit rotation axis k = [vX/s, vY/s, 0]
        val kX = vX / s
        val kY = vY / s

        // k x v
        val kCrossVx = -kY * vz
        val kCrossVy = kX * vz
        val kCrossVz = kX * vy - kY * vx

        // k . v
        val kDotV = kX * vx + kY * vy

        val rotX = vx * c + kCrossVx * s + kX * kDotV * (1.0 - c)
        val rotY = vy * c + kCrossVy * s + kY * kDotV * (1.0 - c)
        val rotZ = vz * c + kCrossVz * s

        return Triple(rotX, rotY, rotZ)
    }
}
