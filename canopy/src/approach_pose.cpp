/**
 * @file approach_pose.cpp
 * @brief Places around a footprint, scored by walk and by how near a side's middle they stand.
 */

#include "canopy/approach_pose.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <opencv2/imgproc.hpp>

namespace canopy
{

namespace
{

/// A place to stand in the footprint's frame, the heading there relative to its axes, and how far
/// round the footprint it is from the middle of a side, m.
struct Candidate
{
    double u;
    double v;
    double yaw;
    double off_middle;
};

}  // namespace

std::optional<Pose2D> approachPose(
    const cv::Mat& cells, const std::vector<float>& travel, const GridGeometry& geometry,
    const Footprint& target, double standoff, const ApproachParams& params)
{
    if (cells.empty() || travel.size() != geometry.cellCount())
    {
        return std::nullopt;
    }
    // Clearance from everything but the target, whose distance the standoff sets: its own cells,
    // a cell or two out of its box, would otherwise rule out its sides.
    cv::Mat others = cells != kOccupied;
    fillFootprint(others, geometry, target, 2.0 * geometry.resolution);
    cv::Mat clearance;
    cv::distanceTransform(others, clearance, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    clearance *= geometry.resolution;

    constexpr double kPi    = std::numbers::pi;
    const double     needed = params.robot_radius + params.margin;
    const double     c      = std::cos(target.yaw);
    const double     s      = std::sin(target.yaw);
    const double     half_x = 0.5 * target.size.x();
    const double     half_y = 0.5 * target.size.y();
    // The target is kept as far off as anything else, and further when asked.
    const double first =
        std::max(needed, std::isfinite(standoff) && standoff > 0.0 ? standoff : params.standoff);

    // A standoff past max_standoff is still tried, as the only one.
    const int rings =
        std::max(0, static_cast<int>((params.max_standoff - first) / params.ring_step + 1e-9));
    const int slides = static_cast<int>(std::max(half_x, half_y) / params.sample_step + 1e-9);
    for (int ring_index = 0; ring_index <= rings; ++ring_index)
    {
        const double           ring = first + (ring_index * params.ring_step);
        std::vector<Candidate> candidates;
        // Square to each side, facing into it, from its middle out along it.
        for (int slide = -slides; slide <= slides; ++slide)
        {
            const double along = slide * params.sample_step;
            if (std::abs(along) <= half_y + 1e-9)
            {
                candidates.push_back({ half_x + ring, along, kPi, std::abs(along) });
                candidates.push_back({ -half_x - ring, along, 0.0, std::abs(along) });
            }
            if (std::abs(along) <= half_x + 1e-9)
            {
                candidates.push_back({ along, half_y + ring, -0.5 * kPi, std::abs(along) });
                candidates.push_back({ along, -half_y - ring, 0.5 * kPi, std::abs(along) });
            }
        }
        // Round each corner, between its two sides' normals, facing the middle.
        const int arc = std::max(2, static_cast<int>(0.5 * kPi * ring / params.sample_step));
        for (const double su : { -1.0, 1.0 })
        {
            for (const double sv : { -1.0, 1.0 })
            {
                const double start = std::atan2(sv, su) - (0.25 * kPi);
                // The half-length of the side square to a normal: an end spans v, a side u.
                const auto half_of = [&](double angle) {
                    return std::abs(std::cos(angle)) > 0.5 ? half_y : half_x;
                };
                for (int i = 1; i < arc; ++i)
                {
                    const double turned = 0.5 * kPi * i / arc;
                    const double angle  = start + turned;
                    const double u      = (su * half_x) + (ring * std::cos(angle));
                    const double v      = (sv * half_y) + (ring * std::sin(angle));
                    const double round  = std::min(
                        half_of(start) + (ring * turned),
                        half_of(start + (0.5 * kPi)) + (ring * ((0.5 * kPi) - turned)));
                    candidates.push_back({ u, v, std::atan2(-v, -u), round });
                }
            }
        }
        // The least walk, each metre off a side's middle counting as off_middle_weight of it.
        std::optional<Pose2D> best;
        double                best_score = std::numeric_limits<double>::infinity();
        for (const Candidate& candidate : candidates)
        {
            const double    x    = target.centre.x() + (c * candidate.u) - (s * candidate.v);
            const double    y    = target.centre.y() + (s * candidate.u) + (c * candidate.v);
            const CellIndex cell = geometry.toCell(x, y);
            if (!geometry.contains(cell) || clearance.at<float>(cell.y, cell.x) < needed)
            {
                continue;
            }
            const double walk  = travel[static_cast<std::size_t>(geometry.index(cell))];
            const double score = walk + (params.off_middle_weight * candidate.off_middle);
            if (std::isfinite(walk) && score < best_score)
            {
                best_score = score;
                best       = Pose2D{ x, y, std::remainder(target.yaw + candidate.yaw, 2.0 * kPi) };
            }
        }
        if (best)
        {
            return best;
        }
    }
    return std::nullopt;
}

}  // namespace canopy
