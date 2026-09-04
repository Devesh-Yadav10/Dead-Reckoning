package com.navigate.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.preference.PreferenceManager
import android.widget.Toast
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.navigate.app.R
import com.navigate.app.bridge.MockNavigationBridge
import com.navigate.app.bridge.NavigationBridge
import com.navigate.app.bridge.RealNavigationBridge
import android.graphics.Color
import com.navigate.app.databinding.ActivityMainBinding
import com.navigate.app.models.Destination
import com.navigate.app.models.GNSSState
import com.navigate.app.models.NavigationState
import com.navigate.app.models.Route
import com.navigate.app.models.RouteRequest
import com.navigate.app.sensors.AndroidLocationAdapter
import com.navigate.app.sensors.AndroidSensorAdapter
import com.navigate.app.services.DestinationSearchService
import com.navigate.app.services.NominatimDestinationSearchService
import com.navigate.app.services.OsrmRouteService
import com.navigate.app.services.RouteService
import kotlinx.coroutines.launch
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private lateinit var sensorAdapter: AndroidSensorAdapter
    private lateinit var locationAdapter: AndroidLocationAdapter

    private val mockBridge = MockNavigationBridge()
    private var realBridge: RealNavigationBridge? = null
    private var activeBridge: NavigationBridge = mockBridge

    private var isLiveMode = true
    private var isBlackoutSimulated = false
    private var isLiveSessionInitialized = false

    var searchService: DestinationSearchService = NominatimDestinationSearchService()
    var routeService: RouteService = OsrmRouteService()
    var selectedDestination: Destination? = null
        private set
    var currentRoute: Route? = null
        private set
    private var destinationMarker: Marker? = null
    private var routeOverlay: Polyline? = null
    private var currentVehicleState: NavigationState? = null
    private var routeGenerationId: Long = 0L

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
                permissions[Manifest.permission.ACCESS_COARSE_LOCATION] == true
        if (granted) {
            locationAdapter.startListening()
        } else {
            Toast.makeText(this, "Location permission required for GNSS fusion", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1. Initialize osmdroid configuration
        Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 2. Setup osmdroid MapView
        setupMapView()

        // 3. Initialize Real ONNX Navigation Bridge (with fallback to Mock)
        try {
            realBridge = RealNavigationBridge.createFromAssets(this)
            activeBridge = realBridge!!
        } catch (e: Exception) {
            e.printStackTrace()
            activeBridge = mockBridge
        }

        // 4. Setup Sensor and Location Adapters
        sensorAdapter = AndroidSensorAdapter(this) { imuSample ->
            if (isLiveMode && isLiveSessionInitialized) {
                activeBridge.processIMU(imuSample)
            }
        }

        locationAdapter = AndroidLocationAdapter(this) { gnssSample ->
            if (isLiveMode) {
                if (!isLiveSessionInitialized) {
                    isLiveSessionInitialized = true
                    val initHeading = gnssSample.bearingDeg ?: 0.0
                    activeBridge.startSession(gnssSample.latitude, gnssSample.longitude, initHeading)
                    runOnUiThread {
                        binding.mapView.controller.setCenter(GeoPoint(gnssSample.latitude, gnssSample.longitude))
                    }
                }
                activeBridge.processGNSS(gnssSample)
            }
        }

        // 5. Connect Bridges to UI updates
        realBridge?.setNavigationStateListener { state ->
            if (isLiveMode) {
                runOnUiThread { updateNavigationUI(state) }
            }
        }
        mockBridge.setNavigationStateListener { state ->
            if (!isLiveMode) {
                runOnUiThread { updateNavigationUI(state) }
            }
        }

        // 6. Setup Mode, Outage, and Search Controls
        setupControlButtons()
        setupSearchUI()

        // 7. Request Location Permissions
        checkLocationPermissions()
    }

    private fun setupMapView() {
        binding.mapView.setTileSource(TileSourceFactory.MAPNIK)
        binding.mapView.setMultiTouchControls(true)
        binding.mapView.controller.setZoom(18.0)
        binding.mapView.controller.setCenter(GeoPoint(51.5074, -0.1278))
    }

    private fun setupSearchUI() {
        binding.btnSearch.setOnClickListener {
            performDestinationSearch()
        }

        binding.etSearchQuery.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                performDestinationSearch()
                true
            } else {
                false
            }
        }

        binding.btnClearDestination.setOnClickListener {
            clearDestination()
        }
    }

    private fun performDestinationSearch() {
        val query = binding.etSearchQuery.text.toString().trim()
        if (query.isEmpty()) {
            binding.tvSearchStatus.text = getString(R.string.hint_search_destination)
            binding.tvSearchStatus.visibility = View.VISIBLE
            binding.layoutSearchResults.visibility = View.GONE
            return
        }

        // Hide keyboard
        val imm = getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager
        imm?.hideSoftInputFromWindow(binding.etSearchQuery.windowToken, 0)

        binding.tvSearchStatus.text = getString(R.string.status_searching)
        binding.tvSearchStatus.visibility = View.VISIBLE
        binding.layoutSearchResults.removeAllViews()
        binding.layoutSearchResults.visibility = View.GONE

        lifecycleScope.launch {
            val result = searchService.search(query)
            result.onSuccess { destinations ->
                if (destinations.isEmpty()) {
                    binding.tvSearchStatus.text = getString(R.string.status_no_results)
                    binding.tvSearchStatus.visibility = View.VISIBLE
                    binding.layoutSearchResults.visibility = View.GONE
                } else {
                    binding.tvSearchStatus.visibility = View.GONE
                    displaySearchResults(destinations)
                }
            }.onFailure { error ->
                binding.tvSearchStatus.text = getString(R.string.status_search_error)
                binding.tvSearchStatus.visibility = View.VISIBLE
                binding.layoutSearchResults.visibility = View.GONE
            }
        }
    }

    private fun displaySearchResults(destinations: List<Destination>) {
        binding.layoutSearchResults.removeAllViews()
        binding.layoutSearchResults.visibility = View.VISIBLE

        for (dest in destinations) {
            val itemView = layoutInflater.inflate(R.layout.item_search_result, binding.layoutSearchResults, false)
            val tvName = itemView.findViewById<TextView>(R.id.tvResultName)
            val tvAddress = itemView.findViewById<TextView>(R.id.tvResultAddress)

            tvName.text = dest.displayName
            tvAddress.text = dest.address

            itemView.setOnClickListener {
                selectDestination(dest)
            }
            binding.layoutSearchResults.addView(itemView)
        }
    }

    fun selectDestination(destination: Destination) {
        selectedDestination = destination
        binding.layoutSearchResults.visibility = View.GONE
        binding.tvSearchStatus.visibility = View.GONE
        binding.etSearchQuery.setText("")

        // Update selected destination banner
        binding.layoutSelectedDestination.visibility = View.VISIBLE
        binding.tvSelectedDestName.text = destination.displayName
        binding.tvSelectedDestCoords.text = String.format("%.6f°, %.6f°", destination.latitude, destination.longitude)

        // Update Map Marker
        if (destinationMarker == null) {
            destinationMarker = Marker(binding.mapView).apply {
                setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                icon = ContextCompat.getDrawable(this@MainActivity, R.drawable.ic_destination_marker)
            }
            binding.mapView.overlays.add(destinationMarker)
        }
        destinationMarker?.position = GeoPoint(destination.latitude, destination.longitude)
        destinationMarker?.title = destination.displayName
        binding.mapView.invalidate()

        // Calculate and visualize OSM route
        calculateRoute(destination)
    }

    fun calculateRoute(destination: Destination) {
        val currentGen = ++routeGenerationId
        val originLat = currentVehicleState?.latitude ?: 51.5074
        val originLon = currentVehicleState?.longitude ?: -0.1278

        binding.tvRouteSummary.text = getString(R.string.status_routing_calculating)
        binding.tvRouteSummary.visibility = View.VISIBLE

        lifecycleScope.launch {
            val request = RouteRequest(
                originLatitude = originLat,
                originLongitude = originLon,
                destinationLatitude = destination.latitude,
                destinationLongitude = destination.longitude
            )
            val result = routeService.getRoute(request)

            // Race guard: ignore if another route was requested or destination cleared
            if (currentGen != routeGenerationId || selectedDestination != destination) {
                return@launch
            }

            result.onSuccess { route ->
                currentRoute = route
                mockBridge.setRoute(route)
                displayRoute(route)
                binding.tvRouteSummary.text = "OSM ROUTE: ${route.summary}"
                binding.tvRouteSummary.visibility = View.VISIBLE
            }.onFailure {
                currentRoute = null
                mockBridge.setRoute(null)
                removeRouteOverlay()
                binding.tvRouteSummary.text = getString(R.string.status_route_unavailable)
                binding.tvRouteSummary.visibility = View.VISIBLE
            }
        }
    }

    fun displayRoute(route: Route) {
        removeRouteOverlay()
        routeOverlay = Polyline(binding.mapView).apply {
            setPoints(route.geometry)
            outlinePaint.color = Color.parseColor("#38BDF8")
            outlinePaint.strokeWidth = 10.0f
        }
        binding.mapView.overlays.add(0, routeOverlay) // Polyline beneath markers
        binding.mapView.invalidate()
    }

    fun removeRouteOverlay() {
        routeOverlay?.let {
            binding.mapView.overlays.remove(it)
            routeOverlay = null
            binding.mapView.invalidate()
        }
    }

    fun clearDestination() {
        ++routeGenerationId
        selectedDestination = null
        currentRoute = null
        mockBridge.setRoute(null)
        binding.layoutSelectedDestination.visibility = View.GONE
        binding.tvRouteSummary.visibility = View.GONE
        destinationMarker?.let {
            binding.mapView.overlays.remove(it)
            destinationMarker = null
        }
        removeRouteOverlay()
        binding.mapView.invalidate()
    }

    private fun setupControlButtons() {
        binding.btnSimulateOutage.setOnClickListener {
            isBlackoutSimulated = !isBlackoutSimulated
            activeBridge.setBlackout(isBlackoutSimulated)
            updateOutageButtonUI()
        }
        updateOutageButtonUI()

        binding.btnLive.setOnClickListener {
            isLiveMode = true
            isBlackoutSimulated = false
            isLiveSessionInitialized = false
            updateOutageButtonUI()
            mockBridge.stopReplay()
            activeBridge = realBridge ?: mockBridge
            sensorAdapter.start()
            locationAdapter.startListening()

            binding.btnLive.backgroundTintList = ContextCompat.getColorStateList(this, R.color.accent)
            binding.btnLive.setTextColor(ContextCompat.getColor(this, R.color.primary_dark))
            binding.btnReplay.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary)
            binding.btnReplay.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
        }

        binding.btnReplay.setOnClickListener {
            isLiveMode = false
            isBlackoutSimulated = false
            isLiveSessionInitialized = false
            updateOutageButtonUI()
            sensorAdapter.stop()
            locationAdapter.stopListening()
            activeBridge.stopSession()
            mockBridge.startReplay()

            binding.btnReplay.backgroundTintList = ContextCompat.getColorStateList(this, R.color.accent)
            binding.btnReplay.setTextColor(ContextCompat.getColor(this, R.color.primary_dark))
            binding.btnLive.backgroundTintList = ContextCompat.getColorStateList(this, R.color.primary)
            binding.btnLive.setTextColor(ContextCompat.getColor(this, R.color.text_primary))
        }
    }

    private fun updateOutageButtonUI() {
        if (isBlackoutSimulated) {
            binding.btnSimulateOutage.text = getString(R.string.btn_restore_gnss)
            binding.btnSimulateOutage.backgroundTintList = ContextCompat.getColorStateList(this, R.color.badge_blue)
        } else {
            binding.btnSimulateOutage.text = getString(R.string.btn_simulate_outage)
            binding.btnSimulateOutage.backgroundTintList = ContextCompat.getColorStateList(this, R.color.badge_red)
        }
    }

    private fun updateNavigationUI(state: NavigationState) {
        currentVehicleState = state

        // A. Update Map View Position & Marker Rotation
        val geoPoint = GeoPoint(state.latitude, state.longitude)
        binding.mapView.controller.animateTo(geoPoint)
        binding.ivCenterMarker.rotation = state.headingDeg.toFloat()

        // B. Update Telemetry HUD
        binding.tvSpeed.text = state.formattedSpeed
        binding.tvHeading.text = state.formattedHeading
        binding.tvLatLon.text = String.format("%.6f°, %.6f°", state.latitude, state.longitude)
        binding.tvLatency.text = String.format("Latency: %.1f ms | 10 Hz Loop", state.latencyMs)

        // C. Update GNSS & Outage Status Badges
        when (state.gnssState) {
            GNSSState.AVAILABLE -> {
                binding.tvGnssStatus.text = getString(R.string.gnss_available)
                binding.tvGnssStatus.setBackgroundResource(R.drawable.bg_badge_green)
                binding.tvModeStatus.text = getString(R.string.mode_normal)
                binding.tvModeStatus.setTextColor(ContextCompat.getColor(this, R.color.text_secondary))
                binding.tvOsmRoad.text = "STANDBY (GNSS Active)"
                binding.tvOsmRoad.setTextColor(ContextCompat.getColor(this, R.color.badge_green))
            }
            GNSSState.BLACKOUT, GNSSState.LOST -> {
                binding.tvGnssStatus.text = getString(R.string.gnss_lost)
                binding.tvGnssStatus.setBackgroundResource(R.drawable.bg_badge_red)
                binding.tvModeStatus.text = getString(R.string.mode_blackout)
                binding.tvModeStatus.setTextColor(ContextCompat.getColor(this, R.color.badge_red))

                if (state.roadConstraintActive && state.matchedRoadName != null) {
                    binding.tvOsmRoad.text = "LOCKED: ${state.matchedRoadName}"
                    binding.tvOsmRoad.setTextColor(ContextCompat.getColor(this, R.color.accent))
                } else {
                    binding.tvOsmRoad.text = "SEARCHING CORRIDOR..."
                    binding.tvOsmRoad.setTextColor(ContextCompat.getColor(this, R.color.badge_orange))
                }
            }
            GNSSState.REACQUIRED -> {
                binding.tvGnssStatus.text = getString(R.string.gnss_reacquired)
                binding.tvGnssStatus.setBackgroundResource(R.drawable.bg_badge_blue)
                binding.tvModeStatus.text = getString(R.string.mode_normal)
                binding.tvModeStatus.setTextColor(ContextCompat.getColor(this, R.color.accent))
                binding.tvOsmRoad.text = "STANDBY (GNSS Active)"
                binding.tvOsmRoad.setTextColor(ContextCompat.getColor(this, R.color.badge_green))
            }
        }
    }

    private fun checkLocationPermissions() {
        val fineGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val coarseGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED

        if (fineGranted || coarseGranted) {
            locationAdapter.startListening()
        } else {
            locationPermissionLauncher.launch(
                arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
            )
        }
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
        if (isLiveMode) {
            sensorAdapter.start()
            locationAdapter.startListening()
        }
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
        sensorAdapter.stop()
        locationAdapter.stopListening()
        mockBridge.stopReplay()
    }

    override fun onDestroy() {
        super.onDestroy()
        activeBridge.reset()
        mockBridge.reset()
        realBridge?.close()
    }
}
