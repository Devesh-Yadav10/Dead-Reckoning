package com.navigate.app.bridge

import com.navigate.app.models.GNSSSample
import com.navigate.app.models.IMUSample
import com.navigate.app.models.NavigationState

/**
 * NavigationBridge — Contract connecting Android sensors / UI to the navigation core.
 */
interface NavigationBridge {
    fun startSession(refLat: Double, refLon: Double, initHeadingDeg: Double = 0.0)
    fun stopSession()
    fun reset()
    fun processIMU(sample: IMUSample)
    fun processGNSS(sample: GNSSSample)
    fun setBlackout(enabled: Boolean)
    fun setNavigationStateListener(listener: (NavigationState) -> Unit)
}

