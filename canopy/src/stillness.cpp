#include "canopy/stillness.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace canopy
{

bool stillOver(
    const std::deque<MotionSample>& samples, double stamp_s, double window_s, double max_speed_mps,
    double max_turn_rate_rad_s)
{
    // Samples just after the stamp count too: odometry and camera stamps are not one clock tick.
    const auto first =
        std::ranges::lower_bound(samples, stamp_s - window_s, {}, &MotionSample::stamp_s);
    const auto last = std::ranges::upper_bound(samples, stamp_s + 0.05, {}, &MotionSample::stamp_s);
    const auto count = std::distance(first, last);
    if (count < 3 || first->stamp_s > stamp_s - (0.8 * window_s))
    {
        return false;
    }
    const MotionSample& at = *std::prev(last);
    double              vx = 0.0;
    double              vy = 0.0;
    double              wz = 0.0;
    for (auto it = first; it != last; ++it)
    {
        const double turn = std::remainder(it->yaw - at.yaw, 2.0 * std::numbers::pi);
        if (std::hypot(it->x - at.x, it->y - at.y) > max_speed_mps * window_s ||
            std::abs(turn) > max_turn_rate_rad_s * window_s)
        {
            return false;
        }
        vx += it->vx;
        vy += it->vy;
        wz += it->wz;
    }
    const auto n = static_cast<double>(count);
    return std::hypot(vx / n, vy / n) <= max_speed_mps && std::abs(wz / n) <= max_turn_rate_rad_s;
}

}  // namespace canopy
