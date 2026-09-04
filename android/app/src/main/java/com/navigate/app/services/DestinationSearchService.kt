package com.navigate.app.services

import com.navigate.app.models.Destination

/**
 * DestinationSearchService — Contract for geocoding and destination search.
 */
interface DestinationSearchService {
    suspend fun search(query: String): Result<List<Destination>>
}

