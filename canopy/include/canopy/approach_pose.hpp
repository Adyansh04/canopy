#ifndef CANOPY__APPROACH_POSE_HPP_
#define CANOPY__APPROACH_POSE_HPP_

/**
 * @file approach_pose.hpp
 * @brief A reachable standing pose facing a target, for Nav2.
 *
 * Candidates stand the standoff away from the target's footprint, square to a side or round a
 * corner facing its centre, reachable and clear of everything else by the robot's radius and a
 * margin. The least walk wins, each metre off a side's middle counting as off_middle_weight metres
 * of it. A wider standoff is tried only when none is clear.
 */

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

#include "canopy/grid.hpp"
#include "canopy/viewpoint_planner.hpp"

namespace canopy
{

struct ApproachParams
{
    double robot_radius = 0.45;  ///< Clearance a standing pose needs, m.
    double margin       = 0.10;  ///< Clearance from any obstacle beyond robot_radius, m.
    double standoff     = 0.60;  ///< Default distance from the footprint's edge, m.
    double max_standoff = 1.80;  ///< Widest standoff tried, m.
    double ring_step    = 0.15;  ///< Standoff increment, m.
    double sample_step  = 0.15;  ///< Spacing of candidates along a side or around a corner, m.
    /// Metres of walk worth standing a metre nearer a side's middle.
    double off_middle_weight = 3.0;
};

/**
 * @brief A reachable pose facing @p target, square to one of its sides when one is free.
 *
 * @param cells     The map's Cell grid on @p geometry.
 * @param travel    Path lengths from the robot, from ViewpointPlanner::travel().
 * @param geometry  Grid placement.
 * @param target    What to face.
 * @param standoff  Distance from the footprint, m, at least robot_radius + margin; 0 uses
 *                  params.standoff.
 * @return The pose, or nothing when no ring out to max_standoff has a reachable clear spot.
 */
std::optional<Pose2D> approachPose(
    const cv::Mat& cells, const std::vector<float>& travel, const GridGeometry& geometry,
    const Footprint& target, double standoff, const ApproachParams& params);

}  // namespace canopy

#endif  // CANOPY__APPROACH_POSE_HPP_
