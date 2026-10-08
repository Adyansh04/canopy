#ifndef CANOPY__ROOM_TYPING_HPP_
#define CANOPY__ROOM_TYPING_HPP_

/**
 * @file room_typing.hpp
 * @brief What kind of room a region is, from the objects in it and its shape.
 *
 * Naive Bayes over the distinct labels found in the room, from a table of how often each label
 * appears in each room type. Scoring by co-occurring objects types distinctive rooms (kitchen,
 * bedroom, bathroom) well and generic ones poorly (Chen et al. 2022), so a type comes with a
 * probability: a describer is asked where it is low, and an operator can overrule it. Long narrow
 * regions with little in them are hallways whatever the table says.
 */

#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace canopy
{

struct RoomTypeTable
{
    std::vector<std::string>                   types;
    std::vector<double>                        priors;
    std::map<std::string, std::vector<double>> likelihood;  ///< label -> P(present | type).
    std::map<std::string, std::string>         synonyms;    ///< Other word -> the table's word.
    double unlisted = 0.08;  ///< P(present | type) for a label the type does not list.

    /// A hallway: narrower than max_width and aspect times longer with at most most_objects kinds
    /// in it, or corridor_aspect times longer whatever is in it. Typed with this probability.
    struct Hallway
    {
        double      max_width       = 2.2;
        double      aspect          = 3.0;
        std::size_t most_objects    = 2;
        double      corridor_aspect = 5.0;
        double      probability     = 0.8;
    } hallway;

    /// Parses the `room_types` table, the `synonyms` and the `hallway` rule of a YAML file; throws
    /// YAML::Exception on bad input.
    static RoomTypeTable fromYaml(const std::string& path);

    /// @p word lower-cased and in the table's words: "Fridge" is "refrigerator".
    [[nodiscard]] std::string canonical(const std::string& word) const;
};

struct RoomTyping
{
    std::string type;  ///< Empty when nothing in the room says anything.
    double      probability = 0.0;
};

/**
 * @brief Types a room.
 *
 * @param table   Likelihoods.
 * @param labels  Labels of the objects in the room; repeats count once.
 * @param length  Long side of the room's minimum-area rectangle, m.
 * @param width   Short side, m.
 */
RoomTyping classifyRoom(
    const RoomTypeTable& table, const std::vector<std::string>& labels, double length, double width);

/**
 * @brief Whether the objects' typing replaces a room's current one.
 *
 * An operator's type stands. A describer is asked only while the objects leave the type in doubt,
 * so its answer stands until they settle on another: at @p settled or more, and at least as sure
 * as the describer was.
 *
 * @param objects  The objects' typing, from classifyRoom.
 * @param current  The room's type and confidence now.
 * @param source   Who typed it: "operator", "describer", "objects", or empty.
 * @param settled  The confidence from which the objects leave no doubt.
 */
bool objectsRetype(
    const RoomTyping& objects, const RoomTyping& current, std::string_view source, double settled);

}  // namespace canopy

#endif  // CANOPY__ROOM_TYPING_HPP_
