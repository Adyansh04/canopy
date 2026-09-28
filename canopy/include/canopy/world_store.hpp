#ifndef CANOPY__WORLD_STORE_HPP_
#define CANOPY__WORLD_STORE_HPP_

/**
 * @file world_store.hpp
 * @brief Saving and loading the world model next to the map it was built on.
 *
 * world.yaml holds what people read (rooms, object labels, names, poses), objects.bin the voxels
 * and embeddings, coverage.bin the camera coverage layers and the occupancy classes they were
 * built on. A world is only reloaded onto the grid it was built on (worldFits()), so a
 * re-mapped building starts clean instead of wearing someone else's objects.
 */

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "canopy/object_map.hpp"

namespace canopy
{

/// What survives about a room; its geometry is re-segmented from the map on load.
struct RoomRecord
{
    std::string id;    ///< "R3".
    std::string name;  ///< "room C", or an operator's name.
    std::string type;
    double      type_confidence = 0.0;
    std::string type_source;
    double      x       = 0.0;  ///< A point inside the room, to find it again, m.
    double      y       = 0.0;
    bool        checked = false;  ///< An operator reviewed it; carried for tools.
    /// Its outline when saved, map frame, for tools: loading re-segments the map instead.
    std::vector<Eigen::Vector2d> outline;
};

struct WorldSnapshot
{
    GridGeometry               geometry;
    std::vector<std::uint8_t>  cells;  ///< Cell classes of the map it was built on.
    std::vector<std::uint8_t>  plan;   ///< The floor plan saved as map.pgm; empty from old saves.
    int                        next_room = 1;
    std::vector<RoomRecord>    rooms;
    std::vector<MappedObject>  objects;
    std::vector<std::uint8_t>  quality;
    std::vector<std::uint8_t>  surface_quality;
    std::vector<std::uint8_t>  flags;
    std::vector<std::uint16_t> structure_hits;  ///< Wall-height LiDAR returns per cell.
    std::vector<std::uint16_t> band_clear;      ///< Rays that crossed the wall band per cell.
};

/**
 * @brief Writes @p snapshot into @p directory, creating it; each file is replaced atomically.
 * @return An empty string on success, else what failed.
 */
std::string saveWorld(const std::string& directory, const WorldSnapshot& snapshot);

/**
 * @brief Whether a saved world belongs on this map: same grid within a centimetre, and nearly
 * every cell classed alike, as the map it was built on or as its floor plan (map.pgm, which is
 * what map_server serves back). A new map does not fit.
 */
bool worldFits(
    const WorldSnapshot& snapshot, const cv::Mat& cells, const GridGeometry& geometry,
    double min_agreement = 0.97);

/**
 * @brief Writes a map as map.pgm and map.yaml, as map_saver would, so the map the world was built
 * on is kept beside it and map_server can serve it back.
 *
 * @param cells Cell classes on @p geometry, CV_8UC1.
 * @return An empty string on success, else what failed.
 */
std::string
saveOccupancy(const std::string& directory, const cv::Mat& cells, const GridGeometry& geometry);

/**
 * @brief Reads a snapshot written by saveWorld().
 * @param error Set to what failed when nothing is returned.
 */
std::optional<WorldSnapshot> loadWorld(const std::string& directory, std::string& error);

/**
 * @brief Moves a saved world out of @p directory into a new subdirectory, @p aside, so a world
 * that does not fit this map is kept rather than saved over.
 * @return An empty string on success, else what failed.
 */
std::string setAsideWorld(const std::string& directory, std::string& aside);

}  // namespace canopy

#endif  // CANOPY__WORLD_STORE_HPP_
