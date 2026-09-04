package com.navigate.app.bridge

import com.navigate.app.models.GNSSSample
import com.navigate.app.models.GNSSState
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState
import com.navigate.app.models.Route
import java.util.Timer
import java.util.TimerTask
import kotlin.math.cos
import kotlin.math.sin

/**
 * MockNavigationBridge — Standalone test/replay bridge for Android MVP.
 *
 * Implements deterministic real-time kinematic dead-reckoning and a rich
 * demonstration replay mode exercising all blackout and OSM constraint states.
 */
class MockNavigationBridge(
    private val motionSource: DemoMotionSource = DemoMotionSource()
) : NavigationBridge {

    private var stateListener: ((NavigationState) -> Unit)? = null

    private var refLat = 51.5074
    private var refLon = -0.1278
    private var curLat = refLat
    private var curLon = refLon
    private var curEastM = 0.0
    private var curNorthM = 0.0
    private var curHeadingDeg = 0.0
    private var curSpeedMs = 0.0
    private var isBlackout = false
    private var manualBlackout: Boolean? = null
    private var reacquisitionTicks = 0
    private var isSessionActive = false

    private var replayTimer: Timer? = null
    var isReplayRunning = false
        private set

    var activeRoute: Route? = null
        private set
    var routeTrajectoryGenerator: RouteTrajectoryGenerator? = null
        private set

    fun setRoute(route: Route?) {
        this.activeRoute = route
        this.routeTrajectoryGenerator = if (route != null && route.geometry.size >= 2) {
            RouteTrajectoryGenerator(
                route = route,
                initialLat = curLat,
                initialLon = curLon,
                initialHeadingDeg = curHeadingDeg
            )
        } else {
            null
        }
    }

    override fun startSession(refLat: Double, refLon: Double, initHeadingDeg: Double) {
        this.refLat = refLat
        this.refLon = refLon
        this.curLat = refLat
        this.curLon = refLon
        this.curEastM = 0.0
        this.curNorthM = 0.0
        this.curHeadingDeg = initHeadingDeg
        this.curSpeedMs = 0.0
        this.isBlackout = false
        this.manualBlackout = null
        this.reacquisitionTicks = 0
        this.isSessionActive = true
        if (activeRoute != null && activeRoute!!.geometry.size >= 2) {
            this.routeTrajectoryGenerator = RouteTrajectoryGenerator(
                route = activeRoute!!,
                initialLat = curLat,
                initialLon = curLon,
                initialHeadingDeg = curHeadingDeg
            )
        }
    }

    override fun stopSession() {
        stopReplay()
        isSessionActive = false
    }

    override fun reset() {
        stopSession()
        curEastM = 0.0
        curNorthM = 0.0
        curHeadingDeg = 0.0
        curSpeedMs = 0.0
        isBlackout = false
        manualBlackout = null
        reacquisitionTicks = 0
        activeRoute = null
        routeTrajectoryGenerator = null
    }

    override fun setBlackout(enabled: Boolean) {
        this.isBlackout = enabled
        this.manualBlackout = enabled
        if (!enabled) {
            this.reacquisitionTicks = 20 // 2.0s at 10Hz
        } else {
            this.reacquisitionTicks = 0
        }
    }

    override fun setNavigationStateListener(listener: (NavigationState) -> Unit) {
        this.stateListener = listener
    }

    override fun processGNSS(sample: GNSSSample) {
        if (!isSessionActive || isReplayRunning) return
        if (!isBlackout) {
            curLat = sample.latitude
            curLon = sample.longitude
        }
    }

    override fun processIMU(sample: IMUSample) {
        if (!isSessionActive || isReplayRunning) return

        val dt = 0.100
        // Simple forward acceleration integration
        curSpeedMs = maxOf(0.0, curSpeedMs + sample.ax * dt)

        val generator = routeTrajectoryGenerator
        if (generator != null && generator.isValid) {
            val trajPt = generator.advance(
                distanceDeltaMeters = curSpeedMs * dt,
                currentSpeedMs = curSpeedMs,
                dt = dt
            )
            curLat = trajPt.latitude
            curLon = trajPt.longitude
            curEastM = trajPt.eastM
            curNorthM = trajPt.northM
            curHeadingDeg = trajPt.headingDeg
            if (trajPt.isCompleted) curSpeedMs = 0.0
        } else {
            curHeadingDeg = (curHeadingDeg + Math.toDegrees(sample.gz * dt)) % 360.0
            if (curHeadingDeg < 0.0) curHeadingDeg += 360.0

            val (dEast, dNorth) = motionSource.calculateDisplacement(curSpeedMs, curHeadingDeg, dt)

            curEastM += dEast
            curNorthM += dNorth

            // ENU to WGS84
            curLat = refLat + (curNorthM / 6371000.0) * (180.0 / Math.PI)
            curLon = refLon + (curEastM / (6371000.0 * cos(Math.toRadians(refLat)))) * (180.0 / Math.PI)
        }

        val gnssState = when {
            isBlackout -> GNSSState.BLACKOUT
            reacquisitionTicks > 0 -> {
                reacquisitionTicks--
                GNSSState.REACQUIRED
            }
            else -> GNSSState.AVAILABLE
        }

        val roadName = when {
            !isBlackout -> null
            activeRoute != null && activeRoute!!.summary.isNotBlank() -> "OSM Route (${activeRoute!!.summary})"
            else -> "Kingsway Corridor (OSM)"
        }

        emitState(
            timestamp = sample.timestamp,
            lat = curLat,
            lon = curLon,
            eastM = curEastM,
            northM = curNorthM,
            speed = curSpeedMs,
            heading = curHeadingDeg,
            gnssState = gnssState,
            blackout = isBlackout,
            roadActive = isBlackout,
            roadName = roadName,
            latency = 1.2
        )
    }

    /**
     * Start deterministic Replay demonstration.
     */
    fun startReplay(onTick: ((NavigationState) -> Unit)? = null) {
        stopReplay()
        startSession(refLat = 51.5074, refLon = -0.1278, initHeadingDeg = motionSource.baseHeadingDeg)
        isReplayRunning = true

        var step = 0

        replayTimer = Timer("ReplayTimer", true)
        replayTimer?.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                val t = step * 0.100
                step++

                // Replay Phases:
                // If manual blackout is set, use it. Otherwise timeline:
                // 0.0 - 5.0s: Normal GNSS
                // 5.0 - 15.0s: GNSS Blackout + OSM Constraint
                // 15.0 - 17.0s: GNSS Reacquired (2.0s transition)
                // 17.0s+: Normal GNSS Available
                val blackoutActive = manualBlackout ?: (t in 5.0..15.0)
                val roadActive = blackoutActive
                val gnssState = when {
                    manualBlackout == true -> GNSSState.BLACKOUT
                    manualBlackout == false -> {
                        if (reacquisitionTicks > 0) {
                            reacquisitionTicks--
                            GNSSState.REACQUIRED
                        } else {
                            GNSSState.AVAILABLE
                        }
                    }
                    t < 5.0 -> GNSSState.AVAILABLE
                    t in 5.0..15.0 -> GNSSState.BLACKOUT
                    t in 15.0..17.0 -> GNSSState.REACQUIRED
                    else -> GNSSState.AVAILABLE
                }

                // Deterministic smooth speed
                val baseSpeed = motionSource.calculateSpeedMs(t)

                val generator = routeTrajectoryGenerator
                if (generator != null && generator.isValid) {
                    val trajPt = generator.advance(
                        distanceDeltaMeters = baseSpeed * 0.100,
                        currentSpeedMs = baseSpeed,
                        dt = 0.100
                    )
                    curLat = trajPt.latitude
                    curLon = trajPt.longitude
                    curEastM = trajPt.eastM
                    curNorthM = trajPt.northM
                    curHeadingDeg = trajPt.headingDeg
                    curSpeedMs = if (trajPt.isCompleted) 0.0 else baseSpeed
                } else {
                    val heading = motionSource.calculateHeadingDeg(t, curHeadingDeg)
                    val (dEast, dNorth) = motionSource.calculateDisplacement(baseSpeed, heading, 0.100)
                    curEastM += dEast
                    curNorthM += dNorth
                    curLat = refLat + (curNorthM / 6371000.0) * (180.0 / Math.PI)
                    curLon = refLon + (curEastM / (6371000.0 * cos(Math.toRadians(refLat)))) * (180.0 / Math.PI)
                    curSpeedMs = baseSpeed
                    curHeadingDeg = heading
                }

                val roadName = when {
                    !roadActive -> null
                    activeRoute != null && activeRoute!!.summary.isNotBlank() -> "OSM Route (${activeRoute!!.summary})"
                    else -> "Kingsway Corridor (OSM)"
                }

                val state = NavigationState(
                    timestamp = t,
                    latitude = curLat,
                    longitude = curLon,
                    posEastM = curEastM,
                    posNorthM = curNorthM,
                    speedMs = curSpeedMs,
                    headingDeg = curHeadingDeg,
                    gnssState = gnssState,
                    blackoutActive = blackoutActive,
                    roadConstraintActive = roadActive,
                    matchedWayId = if (roadActive) 42001L else null,
                    matchedRoadName = roadName,
                    latencyMs = 2.4,
                    diagnostics = mapOf("replay_step" to step, "on_time" to true)
                )

                stateListener?.invoke(state)
                onTick?.invoke(state)
            }
        }, 0L, 100L)
    }

    fun stopReplay() {
        replayTimer?.cancel()
        replayTimer = null
        isReplayRunning = false
    }

    private fun emitState(
        timestamp: Double,
        lat: Double,
        lon: Double,
        eastM: Double,
        northM: Double,
        speed: Double,
        heading: Double,
        gnssState: GNSSState,
        blackout: Boolean,
        roadActive: Boolean,
        roadName: String?,
        latency: Double
    ) {
        val state = NavigationState(
            timestamp = timestamp,
            latitude = lat,
            longitude = lon,
            posEastM = eastM,
            posNorthM = northM,
            speedMs = speed,
            headingDeg = heading,
            gnssState = gnssState,
            blackoutActive = blackout,
            roadConstraintActive = roadActive,
            matchedWayId = if (roadActive) 42001L else null,
            matchedRoadName = roadName,
            latencyMs = latency,
            diagnostics = mapOf("on_time" to true)
        )
        stateListener?.invoke(state)
    }
}

