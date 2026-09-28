/**
 * @file test_world_parts.cpp
 * @brief Grid helpers, room typing, approach poses and persistence.
 */

#include <gmock/gmock.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <opencv2/imgproc.hpp>

#include "canopy/approach_pose.hpp"
#include "canopy/grid.hpp"
#include "canopy/room_typing.hpp"
#include "canopy/world_render.hpp"
#include "canopy/world_store.hpp"

namespace canopy
{
namespace
{

RoomTypeTable table()
{
    return RoomTypeTable::fromYaml(std::string(CANOPY_CONFIG_DIR) + "/room_types.yaml");
}

/// Folded difference of two wall axes, rad: 89 deg and 1 deg are 2 deg apart.
double axisError(double a, double b)
{
    const double d = std::abs(std::remainder(a - b, std::numbers::pi / 2.0));
    return d;
}

/// A 6 x 4 m room's walls, 0.1 m thick, turned by @p yaw about the grid's middle.
cv::Mat turnedRoom(double yaw)
{
    cv::Mat           cells(200, 200, CV_8UC1, cv::Scalar(kFree));
    const cv::Point2f centre(100.0F, 100.0F);
    const auto        corners = [&](double half_x, double half_y) {
        std::vector<cv::Point> out;
        for (const auto& [sx, sy] : { std::pair{ -1, -1 }, { 1, -1 }, { 1, 1 }, { -1, 1 } })
        {
            const double x = sx * half_x;
            const double y = sy * half_y;
            out.emplace_back(
                static_cast<int>(std::lround(centre.x + (x * std::cos(yaw)) - (y * std::sin(yaw)))),
                static_cast<int>(std::lround(centre.y + (x * std::sin(yaw)) + (y * std::cos(yaw)))));
        }
        return out;
    };
    cv::fillConvexPoly(cells, corners(62.0, 42.0), cv::Scalar(kOccupied));
    cv::fillConvexPoly(cells, corners(60.0, 40.0), cv::Scalar(kFree));
    return cells;
}

TEST(Grid, FindsTheWallsAxis)
{
    for (const double degrees : { 0.0, 12.0, 30.0, 45.0, 80.0 })
    {
        const double yaw = degrees * std::numbers::pi / 180.0;
        EXPECT_LT(axisError(dominantAxis(turnedRoom(yaw)), yaw), 2.0 * std::numbers::pi / 180.0)
            << degrees << " deg";
    }
}

TEST(Grid, CarriesALayerOntoAGrownAndShiftedGrid)
{
    const GridGeometry from{ 0.05, 0.0, 0.0, 4, 3 };
    std::vector<int>   layer(from.cellCount());
    for (std::size_t i = 0; i < layer.size(); ++i)
    {
        layer[i] = static_cast<int>(i) + 1;
    }
    // Two cells more on the left and one below: the old cell (0, 0) is now (2, 1).
    const GridGeometry     to{ 0.05, -0.10, -0.05, 7, 5 };
    const std::vector<int> out = remapLayer(layer, from, to, 0);
    EXPECT_EQ(out[static_cast<std::size_t>(to.index(2, 1))], layer[0]);
    EXPECT_EQ(
        out[static_cast<std::size_t>(to.index(5, 3))],
        layer[static_cast<std::size_t>(from.index(3, 2))]);
    EXPECT_EQ(out[static_cast<std::size_t>(to.index(0, 0))], 0);
    EXPECT_EQ(out[static_cast<std::size_t>(to.index(6, 4))], 0);
}

TEST(Grid, FillsAGrownTurnedBox)
{
    // 2 x 1 m at (1, 2) turned a quarter, grown 0.5 m: 2 m along x, 3 m along y.
    const Footprint box{ { 1.0, 2.0 }, { 2.0, 1.0 }, std::numbers::pi / 2.0 };
    const auto      ring = corners(box, 0.5);
    EXPECT_NEAR(ring[0].x(), 0.0, 1e-9);
    EXPECT_NEAR(ring[0].y(), 3.5, 1e-9);
    EXPECT_NEAR(ring[2].x(), 2.0, 1e-9);
    EXPECT_NEAR(ring[2].y(), 0.5, 1e-9);

    const GridGeometry geometry{ 0.1, 0.0, 0.0, 40, 50 };
    cv::Mat            mask(geometry.height, geometry.width, CV_8UC1, cv::Scalar(0));
    fillFootprint(mask, geometry, box, 0.5);
    EXPECT_EQ(mask.at<std::uint8_t>(20, 10), 255);  // The centre.
    EXPECT_EQ(mask.at<std::uint8_t>(34, 10), 255);  // 1.4 m up, inside the 1.5.
    EXPECT_EQ(mask.at<std::uint8_t>(20, 24), 0);    // 1.4 m right, past the 1.0.
}

TEST(RoomTyping, NamesDistinctiveRoomsFromTheirObjects)
{
    const RoomTypeTable types = table();
    EXPECT_EQ(classifyRoom(types, { "refrigerator", "counter", "sink" }, 5.0, 4.0).type, "kitchen");
    EXPECT_EQ(classifyRoom(types, { "bed", "nightstand", "lamp" }, 4.0, 4.0).type, "bedroom");
    EXPECT_EQ(
        classifyRoom(types, { "sofa", "coffee table", "television" }, 6.0, 5.0).type,
        "living room");
    EXPECT_EQ(classifyRoom(types, { "desk", "monitor", "office chair" }, 5.0, 4.0).type, "office");
    const RoomTyping storage = classifyRoom(types, { "cardboard box", "crate", "shelf" }, 4.0, 3.0);
    EXPECT_EQ(storage.type, "storage room");
    EXPECT_GT(storage.probability, 0.5);
}

TEST(RoomTyping, ReadsOtherWordsAsTheTablesOwn)
{
    const RoomTypeTable types = table();
    EXPECT_EQ(types.canonical("Dustbin"), "trash can");
    EXPECT_EQ(types.canonical("fridge"), "refrigerator");
    EXPECT_EQ(types.canonical("lamp"), "lamp");
    EXPECT_EQ(classifyRoom(types, { "fridge", "counter", "dustbin" }, 5.0, 4.0).type, "kitchen");
    // Two words for one kind of object are one piece of evidence, so a hallway stays one. At
    // 4.4 times longer than wide the long-strip rule does not decide it; the object count does.
    EXPECT_EQ(classifyRoom(types, { "bookshelf", "shelf", "bookcase" }, 8.0, 1.8).type, "hallway");
    EXPECT_NE(classifyRoom(types, { "bookshelf", "chair", "desk" }, 8.0, 1.8).type, "hallway");
}

TEST(RoomTyping, ALongEmptyStripIsAHallway)
{
    EXPECT_EQ(classifyRoom(table(), { "potted plant" }, 12.0, 1.8).type, "hallway");
    EXPECT_EQ(classifyRoom(table(), {}, 4.0, 4.0).type, "");
}

TEST(RoomTyping, AStripSixTimesLongerThanWideIsAHallwayWhateverStandsInIt)
{
    const RoomTypeTable types = table();
    EXPECT_EQ(
        classifyRoom(types, { "chair", "trash can", "book", "window" }, 11.9, 2.0).type,
        "hallway");
    // Only three and a half times longer: its objects decide.
    EXPECT_NE(classifyRoom(types, { "chair", "trash can", "book" }, 7.0, 2.0).type, "hallway");
}

TEST(ApproachPose, StandsClearOfATableAndFacesIt)
{
    // A 6 x 6 m room with a 1.2 x 0.8 m table in the middle.
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    cv::Mat            cells(120, 120, CV_8UC1, cv::Scalar(kOccupied));
    cv::rectangle(cells, cv::Point(2, 2), cv::Point(117, 117), cv::Scalar(kFree), cv::FILLED);
    cv::rectangle(cells, cv::Point(48, 52), cv::Point(71, 67), cv::Scalar(kOccupied), cv::FILLED);

    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 1.0, 1.0, 0.0 });
    const Footprint table{ { 3.0, 3.0 }, { 1.2, 0.8 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.0, {});
    ASSERT_TRUE(pose.has_value());
    const double gap = std::hypot(
        std::max(std::abs(pose->x - 3.0) - 0.6, 0.0),
        std::max(std::abs(pose->y - 3.0) - 0.4, 0.0));
    EXPECT_GE(gap, 0.55);
    EXPECT_LE(gap, 0.9);
    // Asked to stand further off than max_standoff, it still finds a spot.
    EXPECT_TRUE(approachPose(cells, planner.travel(), geometry, table, 2.0, {}).has_value());
    // The middle of the end nearest the robot, square to it.
    EXPECT_NEAR(pose->x, 1.8, 1e-9);
    EXPECT_NEAR(pose->y, 3.0, 1e-9);
    EXPECT_NEAR(pose->yaw, 0.0, 1e-9);
}

/// A walled room filling @p geometry, with @p blocks (x, y, width, height, m) solid.
cv::Mat roomWith(const GridGeometry& geometry, const std::vector<cv::Rect2d>& blocks)
{
    cv::Mat cells(geometry.height, geometry.width, CV_8UC1, cv::Scalar(kOccupied));
    cv::rectangle(
        cells,
        cv::Point(2, 2),
        cv::Point(geometry.width - 3, geometry.height - 3),
        cv::Scalar(kFree),
        cv::FILLED);
    for (const cv::Rect2d& block : blocks)
    {
        const CellIndex low = geometry.toCell(block.x, block.y);
        // The far corner is drawn too: its cell is the last inside the block.
        const CellIndex high =
            geometry.toCell(block.x + block.width - 1e-6, block.y + block.height - 1e-6);
        cv::rectangle(
            cells,
            cv::Point(low.x, low.y),
            cv::Point(high.x, high.y),
            cv::Scalar(kOccupied),
            cv::FILLED);
    }
    return cells;
}

TEST(ApproachPose, StandsAtASideWhenTheMarginReachesTheStandoff)
{
    // 0.72 m from anything, standing 0.72 m off: the table itself must not rule its sides out.
    // The map shows it a cell bigger than its box, as the floor plan does furniture.
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    const cv::Mat      cells = roomWith(geometry, { { 2.35, 2.55, 1.3, 0.9 } });
    ViewpointPlanner   planner;
    planner.prepare(cells, geometry, { 1.0, 1.0, 0.0 });
    ApproachParams params;
    params.standoff = 0.72;
    params.margin   = 0.27;
    const Footprint table{ { 3.0, 3.0 }, { 1.2, 0.8 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.0, params);
    ASSERT_TRUE(pose.has_value());
    EXPECT_NEAR(pose->x, 3.0 - 0.6 - 0.72, 1e-9);  // The middle of the near end, not a corner.
    EXPECT_NEAR(pose->y, 3.0, 1e-9);
}

TEST(ApproachPose, TakesTheMiddleOfAFreeSideWhenChairsFillTheNearOne)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    // A 2 x 1.2 m table, a chair at the middle of each long side.
    const cv::Mat cells = roomWith(
        geometry,
        { { 2.0, 2.4, 2.0, 1.2 }, { 2.75, 1.9, 0.5, 0.5 }, { 2.75, 3.6, 0.5, 0.5 } });
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 3.0, 0.8, 0.0 });
    const Footprint table{ { 3.0, 3.0 }, { 2.0, 1.2 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.0, {});
    ASSERT_TRUE(pose.has_value());
    EXPECT_NEAR(std::abs(pose->x - 3.0), 1.6, 1e-9);  // An end's middle...
    EXPECT_NEAR(pose->y, 3.0, 1e-9);
    EXPECT_NEAR(
        std::remainder(pose->yaw - (pose->x > 3.0 ? std::numbers::pi : 0.0), 2.0 * std::numbers::pi),
        0.0,
        1e-9);
}

TEST(ApproachPose, SlidesAlongTheFreeSideWhenItsMiddleIsTaken)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    // A counter against the north wall, cupboards at both ends, a stool at its middle.
    const cv::Mat cells = roomWith(
        geometry,
        { { 2.0, 5.1, 2.0, 0.6 },
          { 1.4, 5.1, 0.6, 0.6 },
          { 4.0, 5.1, 0.6, 0.6 },
          { 2.8, 4.3, 0.4, 0.4 } });
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 3.0, 1.0, 0.0 });
    const Footprint counter{ { 3.0, 5.4 }, { 2.0, 0.6 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, counter, 0.0, {});
    ASSERT_TRUE(pose.has_value());
    // In front of it, square to it, as near its middle as the stool allows.
    EXPECT_NEAR(pose->y, 4.5, 1e-9);
    EXPECT_GT(std::abs(pose->x - 3.0), 0.7);
    EXPECT_LT(std::abs(pose->x - 3.0), 1.0);
    EXPECT_NEAR(pose->yaw, 0.5 * std::numbers::pi, 1e-9);
}

TEST(ApproachPose, RoundsACornerOnlyWhenEverySideIsBlocked)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    // A table with a rail a metre off each side, and gaps at the corners.
    const cv::Mat cells = roomWith(
        geometry,
        { { 2.4, 2.6, 1.2, 0.8 },
          { 2.4, 4.35, 1.2, 0.1 },
          { 2.4, 1.55, 1.2, 0.1 },
          { 4.55, 2.6, 0.1, 0.8 },
          { 1.35, 2.6, 0.1, 0.8 } });
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 1.0, 1.0, 0.0 });
    const Footprint table{ { 3.0, 3.0 }, { 1.2, 0.8 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.0, {});
    ASSERT_TRUE(pose.has_value());
    EXPECT_GT(std::abs(pose->x - 3.0), 0.6);  // Past both sides' ends: a corner...
    EXPECT_GT(std::abs(pose->y - 3.0), 0.4);
    EXPECT_NEAR(  // ...facing the middle.
        std::remainder(pose->yaw - std::atan2(3.0 - pose->y, 3.0 - pose->x), 2.0 * std::numbers::pi),
        0.0,
        1e-9);
}

TEST(ApproachPose, KeepsItsMarginFromAChairWhenAskedToStandCloser)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    // A chair 0.8 m off the table's near end.
    const cv::Mat    cells = roomWith(geometry, { { 2.4, 2.6, 1.2, 0.8 }, { 1.4, 2.8, 0.2, 0.4 } });
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 1.0, 1.0, 0.0 });
    const Footprint table{ { 3.0, 3.0 }, { 1.2, 0.8 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.01, {});
    ASSERT_TRUE(pose.has_value());
    const double to_table = std::hypot(
        std::max(std::abs(pose->x - 3.0) - 0.6, 0.0),
        std::max(std::abs(pose->y - 3.0) - 0.4, 0.0));
    const double to_chair = std::hypot(
        std::max(std::abs(pose->x - 1.5) - 0.1, 0.0),
        std::max(std::abs(pose->y - 3.0) - 0.2, 0.0));
    EXPECT_GE(to_table, 0.55 - 1e-9);  // robot_radius + margin, for the target too...
    EXPECT_GE(to_chair, 0.55);         // ...and everything else.
}

TEST(ApproachPose, TakesASidesMiddleOfARotatedTable)
{
    // Turned 20 degrees, and drawn a cell bigger than its box, as the floor plan draws furniture.
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    cv::Mat            cells = roomWith(geometry, {});
    const Footprint    table{ { 3.0, 3.0 }, { 1.4, 0.8 }, 20.0 * std::numbers::pi / 180.0 };
    cv::Mat            solid(cells.size(), CV_8UC1, cv::Scalar(0));
    fillFootprint(solid, geometry, table, geometry.resolution);
    cells.setTo(kOccupied, solid);
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 1.0, 3.0, 0.0 });
    ApproachParams params;
    params.standoff = 0.72;
    params.margin   = 0.27;
    const auto pose = approachPose(cells, planner.travel(), geometry, table, 0.0, params);
    ASSERT_TRUE(pose.has_value());
    const double dx = pose->x - 3.0;
    const double dy = pose->y - 3.0;
    const double u  = (std::cos(table.yaw) * dx) + (std::sin(table.yaw) * dy);
    const double v  = (-std::sin(table.yaw) * dx) + (std::cos(table.yaw) * dy);
    EXPECT_NEAR(u, -0.7 - 0.72, 1e-9);  // The middle of the end facing the robot.
    EXPECT_NEAR(v, 0.0, 1e-9);
    EXPECT_NEAR(std::remainder(pose->yaw - table.yaw, 2.0 * std::numbers::pi), 0.0, 1e-9);
}

TEST(ApproachPose, TakesANearSpotBesideAStoolOverAFarSidesMiddle)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 120, 120 };
    // A 2 m table set in a wall across the room, whose far side is a walk round the wall's east
    // end; a stool at the middle of its near side.
    const cv::Mat cells = roomWith(
        geometry,
        { { 2.0, 2.6, 2.0, 0.8 },
          { 0.1, 2.95, 1.9, 0.1 },
          { 4.0, 2.95, 1.0, 0.1 },
          { 2.8, 2.1, 0.4, 0.4 } });
    ViewpointPlanner planner;
    planner.prepare(cells, geometry, { 3.0, 1.0, 0.0 });
    const Footprint table{ { 3.0, 3.0 }, { 2.0, 0.8 }, 0.0 };
    const auto      pose = approachPose(cells, planner.travel(), geometry, table, 0.0, {});
    ASSERT_TRUE(pose.has_value());
    EXPECT_NEAR(pose->y, 2.0, 1e-9);  // The near side, beside the stool...
    EXPECT_GT(std::abs(pose->x - 3.0), 0.6);
    EXPECT_NEAR(pose->yaw, 0.5 * std::numbers::pi, 1e-9);  // ...square to it.
}

TEST(WorldStore, RoundTripsRoomsObjectsAndCoverage)
{
    WorldSnapshot snapshot;
    snapshot.geometry  = { 0.05, -2.0, 1.5, 3, 1 };
    snapshot.cells     = { kFree, kOccupied, kUnknown };
    snapshot.next_room = 4;
    snapshot.voxel     = 0.04;
    snapshot.rooms.push_back({ "R2", "room B", "office", 0.7, "objects", 3.5, -1.25 });
    // What a map editor adds: a reviewed room with its outline, an operator's label and box.
    snapshot.rooms.back().checked = true;
    snapshot.rooms.back().outline = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 1.0, 1.5 } };
    MappedObject object;
    object.id              = 12;
    object.votes           = { { "dustbin", 2.5F }, { "bucket", 0.4F } };
    object.name            = "metal bin";
    object.caption         = "A grey metal dustbin in the corner.";
    object.voxels          = { 5, 9, 1024 };
    object.embedding       = { 0.6F, 0.8F };
    object.embedding_count = 3;
    object.observations    = 7;
    object.state           = ObjectState::kStale;
    object.best_view_score = 900.0;
    object.operator_label  = "trash can";
    object.checked         = true;
    object.box_pinned      = true;
    object.removed_by      = "clean-up: the floor";
    object.box_centre      = { 1.0, 2.0 };
    object.box_size        = { 0.3, 0.4 };
    object.box_yaw         = 0.5;
    snapshot.objects.push_back(object);
    // One world.yaml names with nothing in objects.bin, as a save cut short leaves it.
    MappedObject torn;
    torn.id    = 13;
    torn.votes = { { "sofa", 1.0F } };
    snapshot.objects.push_back(torn);
    snapshot.plan            = { kFree, kOccupied, kOccupied };
    snapshot.quality         = { 0, 128, 255 };
    snapshot.surface_quality = { 1, 2, 3 };
    snapshot.flags           = { 0, 1, 0 };
    snapshot.structure_hits  = { 0, 900, 3 };
    snapshot.band_clear      = { 7, 0, 65535 };

    const std::string directory =
        (std::filesystem::temp_directory_path() / "canopy_world_store_test").string();
    std::filesystem::remove_all(directory);
    ASSERT_EQ(saveWorld(directory, snapshot), "");

    std::string error;
    const auto  loaded = loadWorld(directory, error);
    ASSERT_TRUE(loaded.has_value()) << error;
    EXPECT_EQ(loaded->geometry, snapshot.geometry);
    EXPECT_EQ(loaded->cells, snapshot.cells);
    const cv::Mat same  = (cv::Mat_<std::uint8_t>(1, 3) << kFree, kOccupied, kUnknown);
    const cv::Mat other = (cv::Mat_<std::uint8_t>(1, 3) << kOccupied, kOccupied, kFree);
    EXPECT_TRUE(worldFits(*loaded, same, snapshot.geometry));
    EXPECT_FALSE(worldFits(*loaded, other, snapshot.geometry));
    // Its floor plan, as map_server serves map.pgm back, fits as well.
    const cv::Mat served = (cv::Mat_<std::uint8_t>(1, 3) << kFree, kOccupied, kOccupied);
    EXPECT_TRUE(worldFits(*loaded, served, snapshot.geometry));
    EXPECT_EQ(loaded->next_room, 4);
    EXPECT_DOUBLE_EQ(loaded->voxel, 0.04);
    ASSERT_EQ(loaded->rooms.size(), 1U);
    EXPECT_EQ(loaded->rooms[0].name, "room B");
    EXPECT_DOUBLE_EQ(loaded->rooms[0].y, -1.25);
    EXPECT_TRUE(loaded->rooms[0].checked);
    ASSERT_EQ(loaded->rooms[0].outline.size(), 3U);
    EXPECT_DOUBLE_EQ(loaded->rooms[0].outline[2].y(), 1.5);
    ASSERT_EQ(loaded->objects.size(), 1U);  // Not the torn one.
    const MappedObject& back = loaded->objects[0];
    EXPECT_EQ(back.id, 12);
    EXPECT_EQ(back.label(), "trash can");
    EXPECT_FLOAT_EQ(back.votes.at("dustbin"), 2.5F);
    EXPECT_TRUE(back.checked);
    EXPECT_TRUE(back.box_pinned);
    EXPECT_EQ(back.removed_by, object.removed_by);
    EXPECT_DOUBLE_EQ(back.box_centre.y(), 2.0);
    EXPECT_DOUBLE_EQ(back.box_size.y(), 0.4);
    EXPECT_DOUBLE_EQ(back.box_yaw, 0.5);
    EXPECT_EQ(back.caption, object.caption);
    EXPECT_EQ(back.voxels, object.voxels);
    EXPECT_EQ(back.embedding, object.embedding);
    EXPECT_EQ(back.embedding_count, 3);
    EXPECT_EQ(back.state, ObjectState::kStale);
    EXPECT_DOUBLE_EQ(back.best_view_score, 900.0);
    EXPECT_EQ(loaded->structure_hits, snapshot.structure_hits);
    EXPECT_EQ(loaded->band_clear, snapshot.band_clear);

    // Set aside rather than saved over: the directory is empty of it, the copy loads.
    std::string aside;
    ASSERT_EQ(setAsideWorld(directory, aside), "");
    EXPECT_FALSE(loadWorld(directory, error).has_value());
    const auto kept = loadWorld(aside, error);
    ASSERT_TRUE(kept.has_value()) << error;
    EXPECT_EQ(kept->objects.size(), 1U);
    std::filesystem::remove_all(directory);
}

TEST(WorldStore, StampsTheWorldsLastWrite)
{
    const std::string directory =
        (std::filesystem::temp_directory_path() / "canopy_world_stamp_test").string();
    std::filesystem::remove_all(directory);
    EXPECT_FALSE(worldStamp(directory).has_value());
    WorldSnapshot snapshot;
    snapshot.geometry        = { 0.05, 0.0, 0.0, 1, 1 };
    snapshot.cells           = { kFree };
    snapshot.plan            = { kFree };
    snapshot.quality         = { 0 };
    snapshot.surface_quality = { 0 };
    snapshot.flags           = { 0 };
    snapshot.structure_hits  = { 0 };
    snapshot.band_clear      = { 0 };
    ASSERT_EQ(saveWorld(directory, snapshot), "");
    const auto written = worldStamp(directory);
    ASSERT_TRUE(written.has_value());
    EXPECT_EQ(worldStamp(directory), written);
    // A map editor's save moves it on, which is how the node knows not to save over it; so does
    // one landing in the same clock tick, since every save renames a new file into place.
    const std::filesystem::path yaml = std::filesystem::path(directory) / "world.yaml";
    std::filesystem::last_write_time(
        yaml,
        std::filesystem::last_write_time(yaml) + std::chrono::seconds(5));
    EXPECT_NE(worldStamp(directory), written);
    const auto touched = worldStamp(directory);
    ASSERT_EQ(saveWorld(directory, snapshot), "");
    std::filesystem::last_write_time(yaml, std::filesystem::last_write_time(yaml));
    EXPECT_NE(worldStamp(directory)->inode, touched->inode);
}

TEST(WorldStore, ReportsAMissingWorld)
{
    std::string error;
    EXPECT_FALSE(loadWorld("/nonexistent/canopy_world", error).has_value());
    EXPECT_FALSE(error.empty());
}

TEST(Grid, SettlesThePocketsTheMapEncloses)
{
    cv::Mat cells(40, 60, CV_8UC1, cv::Scalar(kUnknown));
    cv::rectangle(cells, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(kOccupied), cv::FILLED);
    cv::rectangle(cells, cv::Point(6, 6), cv::Point(53, 33), cv::Scalar(kFree), cv::FILLED);
    // A sofa whose inside no beam reached.
    cv::rectangle(cells, cv::Point(20, 15), cv::Point(30, 20), cv::Scalar(kOccupied), cv::FILLED);
    cv::rectangle(cells, cv::Point(21, 16), cv::Point(29, 19), cv::Scalar(kUnknown), cv::FILLED);
    // A partition with a stretch no beam hit, and a gap between beams out on the floor.
    cv::line(cells, cv::Point(40, 6), cv::Point(40, 33), cv::Scalar(kOccupied));
    cv::line(cells, cv::Point(40, 20), cv::Point(40, 22), cv::Scalar(kUnknown));
    cells.at<std::uint8_t>(10, 12) = kUnknown;

    const cv::Mat settled = settleEnclosedUnknown(cells);
    EXPECT_EQ(settled.at<std::uint8_t>(0, 0), kUnknown);     // The outside.
    EXPECT_EQ(settled.at<std::uint8_t>(17, 25), kOccupied);  // Inside the sofa.
    EXPECT_EQ(settled.at<std::uint8_t>(21, 40), kOccupied);  // The partition, closed.
    EXPECT_EQ(settled.at<std::uint8_t>(10, 12), kFree);      // The floor, filled.
}

TEST(Grid, ClosesTheWallBehindACounterButNotADoorway)
{
    // A room whose right wall the scan band only sees above a doorway's height, and a counter
    // standing on its top wall, its back and the wall behind it never seen.
    cv::Mat cells(40, 60, CV_8UC1, cv::Scalar(kUnknown));
    cv::rectangle(cells, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(kOccupied), cv::FILLED);
    cv::rectangle(cells, cv::Point(6, 6), cv::Point(53, 33), cv::Scalar(kFree), cv::FILLED);
    cv::rectangle(cells, cv::Point(20, 3), cv::Point(40, 8), cv::Scalar(kUnknown), cv::FILLED);
    cv::line(cells, cv::Point(20, 9), cv::Point(40, 9), cv::Scalar(kOccupied));
    cv::rectangle(cells, cv::Point(54, 15), cv::Point(59, 20), cv::Scalar(kFree), cv::FILLED);
    // Wall-height hits: the whole outline, the lintel over the doorway included.
    cv::Mat walls(cells.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(walls, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(255));

    // A wardrobe on the bottom wall hides it at every height: no hits there either.
    cv::rectangle(cells, cv::Point(20, 30), cv::Point(35, 37), cv::Scalar(kUnknown), cv::FILLED);
    cv::line(cells, cv::Point(20, 30), cv::Point(35, 30), cv::Scalar(kOccupied));
    cv::line(walls, cv::Point(20, 34), cv::Point(35, 34), cv::Scalar(0));

    const cv::Mat plan = completeMap(cells, walls, 0.0, { 20, 12, 3 });
    EXPECT_EQ(plan.at<std::uint8_t>(5, 30), kOccupied);   // The wall behind the counter.
    EXPECT_EQ(plan.at<std::uint8_t>(7, 30), kOccupied);   // The counter, solid to the wall.
    EXPECT_EQ(plan.at<std::uint8_t>(34, 27), kOccupied);  // The wall behind the wardrobe.
    EXPECT_EQ(plan.at<std::uint8_t>(32, 27), kOccupied);  // The wardrobe, solid.
    EXPECT_EQ(plan.at<std::uint8_t>(17, 54), kFree);      // The doorway, lintel or not.
    EXPECT_EQ(plan.at<std::uint8_t>(3, 30), kUnknown);    // Outside, open to the edge.
    EXPECT_EQ(plan.at<std::uint8_t>(37, 27), kUnknown);   // And below the room.
}

TEST(Grid, ClosesACrackTheBeamsCutInAWallButNotADoorway)
{
    // A room whose top wall the scan marked free for three cells, and a doorway six wide.
    cv::Mat cells(40, 60, CV_8UC1, cv::Scalar(kUnknown));
    cv::rectangle(cells, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(kOccupied), cv::FILLED);
    cv::rectangle(cells, cv::Point(6, 6), cv::Point(53, 33), cv::Scalar(kFree), cv::FILLED);
    cv::line(cells, cv::Point(30, 5), cv::Point(32, 5), cv::Scalar(kFree));
    cv::rectangle(cells, cv::Point(54, 15), cv::Point(59, 20), cv::Scalar(kFree), cv::FILLED);
    const cv::Mat walls(cells.size(), CV_8UC1, cv::Scalar(0));

    const cv::Mat plan = completeMap(cells, walls, 0.0, { 20, 12, 3, 4 });
    EXPECT_EQ(plan.at<std::uint8_t>(5, 31), kOccupied);  // The crack.
    EXPECT_EQ(plan.at<std::uint8_t>(17, 54), kFree);     // The doorway.
    EXPECT_EQ(plan.at<std::uint8_t>(10, 31), kFree);     // The room.
}

TEST(Grid, ClosesFurnitureAtTheMapsEdgeWhateverTheWallsYaw)
{
    // A fridge in a room's corner, its back at the grid's edge: the wall behind it is off the map,
    // so only the run between the wall and the fridge's side closes it.
    cv::Mat cells(40, 60, CV_8UC1, cv::Scalar(kFree));
    cv::line(cells, cv::Point(1, 0), cv::Point(1, 39), cv::Scalar(kOccupied));
    cv::rectangle(cells, cv::Point(2, 0), cv::Point(9, 7), cv::Scalar(kUnknown), cv::FILLED);
    cv::line(cells, cv::Point(2, 8), cv::Point(10, 8), cv::Scalar(kOccupied));
    cv::line(cells, cv::Point(10, 0), cv::Point(10, 8), cv::Scalar(kOccupied));
    const cv::Mat walls(cells.size(), CV_8UC1, cv::Scalar(0));
    for (const double yaw : { 0.0, 1.45 })
    {
        EXPECT_EQ(completeMap(cells, walls, yaw, { 20, 12, 3 }).at<std::uint8_t>(3, 5), kOccupied)
            << yaw;
    }
}

TEST(Grid, KeepsTheFloorBehindAWardrobeFreeButFillsWhatTheCameraMapped)
{
    // A room with a wardrobe in its top right corner: the scan saw its west and south faces, and
    // past its corner some of the floor between it and the top wall; the rest of that floor never.
    cv::Mat cells(40, 60, CV_8UC1, cv::Scalar(kUnknown));
    cv::rectangle(cells, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(kOccupied), cv::FILLED);
    cv::rectangle(cells, cv::Point(6, 6), cv::Point(53, 33), cv::Scalar(kFree), cv::FILLED);
    cv::rectangle(cells, cv::Point(45, 6), cv::Point(53, 21), cv::Scalar(kUnknown), cv::FILLED);
    cv::line(cells, cv::Point(44, 12), cv::Point(44, 22), cv::Scalar(kOccupied));
    cv::line(cells, cv::Point(44, 22), cv::Point(53, 22), cv::Scalar(kOccupied));
    // An armchair out on the floor, seen from the north and east only: floor all round the rest.
    cv::rectangle(cells, cv::Point(20, 20), cv::Point(26, 25), cv::Scalar(kUnknown), cv::FILLED);
    cv::line(cells, cv::Point(20, 19), cv::Point(27, 19), cv::Scalar(kOccupied));
    cv::line(cells, cv::Point(27, 19), cv::Point(27, 25), cv::Scalar(kOccupied));
    cv::Mat walls(cells.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(walls, cv::Point(5, 5), cv::Point(54, 34), cv::Scalar(255));
    // The camera mapped the armchair, and the wardrobe from its faces.
    cv::Mat mapped(cells.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(mapped, cv::Point(19, 18), cv::Point(28, 26), cv::Scalar(255), cv::FILLED);
    cv::rectangle(mapped, cv::Point(43, 11), cv::Point(54, 23), cv::Scalar(255), cv::FILLED);

    const cv::Mat plan = completeMap(cells, walls, 0.0, { 20, 12, 3 }, mapped);
    EXPECT_EQ(plan.at<std::uint8_t>(17, 49), kOccupied);  // The wardrobe, solid to the wall.
    EXPECT_EQ(plan.at<std::uint8_t>(8, 49), kFree);       // The floor behind it: floor.
    EXPECT_EQ(plan.at<std::uint8_t>(22, 23), kOccupied);  // The armchair's underside.
    EXPECT_EQ(completeMap(cells, walls, 0.0, { 20, 12, 3 }).at<std::uint8_t>(22, 23), kFree);
}

TEST(WorldStore, SavesTheMapAsMapServerReadsIt)
{
    // 3 x 2: free, occupied, unknown on the bottom row; unknown, free, occupied on the top.
    const GridGeometry geometry{ 0.05, -1.0, 2.0, 3, 2 };
    const cv::Mat      cells =
        (cv::Mat_<std::uint8_t>(2, 3) << kFree, kOccupied, kUnknown, kUnknown, kFree, kOccupied);
    const std::string directory =
        (std::filesystem::temp_directory_path() / "canopy_map_save").string();
    ASSERT_EQ(saveOccupancy(directory, cells, geometry), "");

    std::ifstream pgm(directory + "/map.pgm", std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(pgm)), std::istreambuf_iterator<char>());
    const std::string header = "P5\n3 2\n255\n";
    ASSERT_EQ(bytes.size(), header.size() + 6);
    EXPECT_EQ(bytes.substr(0, header.size()), header);
    // The file's first row is the map's top one.
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 0]), 205);
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 1]), 254);
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 2]), 0);
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 3]), 254);
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 4]), 0);
    EXPECT_EQ(static_cast<unsigned char>(bytes[header.size() + 5]), 205);

    std::ifstream     yaml_file(directory + "/map.yaml");
    const std::string yaml(
        (std::istreambuf_iterator<char>(yaml_file)),
        std::istreambuf_iterator<char>());
    EXPECT_THAT(yaml, ::testing::HasSubstr("origin: [-1, 2, 0]"));
    EXPECT_THAT(yaml, ::testing::HasSubstr("free_thresh: 0.196"));
    std::filesystem::remove_all(directory);
}

TEST(WorldRender, KeepsLabelsApart)
{
    // A mug and a bowl on a table: all three labels would sit on the table's middle.
    const std::vector<cv::Rect>  boxes{ { 40, 40, 80, 40 }, { 70, 55, 8, 8 }, { 84, 55, 8, 8 } };
    const std::vector<cv::Size>  sizes{ { 50, 10 }, { 24, 10 }, { 26, 10 } };
    const std::vector<cv::Point> corners = placeLabels(boxes, sizes, {});
    ASSERT_EQ(corners.size(), 3U);
    EXPECT_EQ(corners[0], cv::Point(55, 55));  // The table keeps its middle.
    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        for (std::size_t j = i + 1; j < corners.size(); ++j)
        {
            EXPECT_EQ((cv::Rect(corners[i], sizes[i]) & cv::Rect(corners[j], sizes[j])).area(), 0)
                << i << " and " << j;
        }
    }
}

TEST(WorldRender, TintsRoomsAndOutlinesObjects)
{
    const GridGeometry geometry{ 0.05, 0.0, 0.0, 40, 20 };
    cv::Mat            cells(geometry.height, geometry.width, CV_8UC1, cv::Scalar(kFree));
    cv::Mat            labels(geometry.height, geometry.width, CV_32S, cv::Scalar(1));
    labels(cv::Rect(20, 0, 20, 20)) = 2;
    const cv::Mat image             = renderWorld(
        cells,
        geometry,
        labels,
        { { "R1 office", 0.5, 0.5 }, { "R2 bedroom", 1.5, 0.5 } },
        { { { { 1.0, 0.5 }, { 0.6, 0.4 }, 0.0 }, "desk" } },
        4);
    ASSERT_EQ(image.cols, 160);
    ASSERT_EQ(image.rows, 80);
    // Two rooms, two tints; neither is the free-space white.
    const auto& left  = image.at<cv::Vec3b>(75, 5);
    const auto& right = image.at<cv::Vec3b>(75, 155);
    EXPECT_NE(left, right);
    EXPECT_NE(left, cv::Vec3b(255, 255, 255));
    // The desk's outline crosses its box's left edge, 0.7 m in: pixel column 56.
    bool outlined = false;
    for (int x = 54; x <= 58; ++x)
    {
        const auto& pixel = image.at<cv::Vec3b>(40, x);
        outlined          = outlined || (pixel[2] > pixel[0] + 60);
    }
    EXPECT_TRUE(outlined);
}

}  // namespace
}  // namespace canopy
