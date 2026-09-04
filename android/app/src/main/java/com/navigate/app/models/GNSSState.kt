package com.navigate.app.models

/**
 * Lifecycle states of GNSS positioning and outage handling.
 */
enum class GNSSState {
    AVAILABLE,
    LOST,
    BLACKOUT,
    REACQUIRED
}

