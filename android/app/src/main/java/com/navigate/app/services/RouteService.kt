package com.navigate.app.services

import com.navigate.app.models.Route
import com.navigate.app.models.RouteRequest

/**
 * RouteService — Contract for acquiring routes from routing providers.
 */
interface RouteService {
    suspend fun getRoute(request: RouteRequest): Result<Route>
}

