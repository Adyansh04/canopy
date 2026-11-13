#pragma once

#include <deque>

namespace canopy
{

/// One odometry sample: when it was taken, where the base was and how it was moving.
struct MotionSample
{
    double stamp_s = 0.0;
    double x       = 0.0;  ///< m, odometry frame
    double y       = 0.0;  ///< m, odometry frame
    double yaw     = 0.0;  ///< rad
    double vx      = 0.0;  ///< m/s, base frame
    double vy      = 0.0;  ///< m/s, base frame
    double wz      = 0.0;  ///< rad/s
};

/**
 * @brief Whether the base was still over the window before a frame.
 *
 * Judged on the window as a whole, so a base that sways where it stands, as a legged robot's does,
 * still counts: the mean velocity, and the furthest the base strayed from its pose at the frame,
 * must both stay within the limits. Walking or turning moves the base past them within the window.
 *
 * @param samples Odometry, oldest first.
 * @param stamp_s The frame's stamp.
 * @param window_s How long the base must have been still before the frame.
 * @param max_speed_mps Mean speed limit; the base may stray max_speed_mps * window_s metres from
 *        its pose at the frame.
 * @param max_turn_rate_rad_s Mean turn-rate limit; the base may turn max_turn_rate_rad_s * window_s.
 * @return False as well when the samples do not cover the window.
 */
[[nodiscard]] bool stillOver(
    const std::deque<MotionSample>& samples, double stamp_s, double window_s, double max_speed_mps,
    double max_turn_rate_rad_s);

}  // namespace canopy
