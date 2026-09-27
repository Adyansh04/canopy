#ifndef CANOPY_PERCEPTION__DEPTH_HISTORY_HPP_
#define CANOPY_PERCEPTION__DEPTH_HISTORY_HPP_

/**
 * @file depth_history.hpp
 * @brief Keeps recent frames so a late mask can be paired with the frame it was cut from.
 *
 * The detector takes seconds; the newest frame would put the object where it no longer is.
 */

#include <cstddef>
#include <deque>
#include <sensor_msgs/msg/image.hpp>

namespace canopy_perception
{

class DepthHistory
{
public:
    /**
     * @param history_s   How far back frames are kept. Sized by the detector's worst latency.
     * @param tolerance_s How closely a frame's stamp must match the mask's.
     * @param max_frames  Cap on frames held regardless of the window, 0 for none. Bounds memory
     *                    on a fast colour stream.
     * @param min_spacing_s A frame this soon after the newest held is not kept, so max_frames
     *                    spans the window on a fast camera too; 0 keeps every frame.
     */
    DepthHistory(
        double history_s, double tolerance_s, std::size_t max_frames = 0,
        double min_spacing_s = 0.0);

    /// Stores a frame unless it comes within min_spacing_s of the newest held, then drops the
    /// oldest past @p history_s behind it or past @p max_frames. Returns whether it was kept.
    bool push(sensor_msgs::msg::Image::ConstSharedPtr frame);

    /// The frame captured at @p stamp_s, or nullptr when none is within the tolerance.
    [[nodiscard]] sensor_msgs::msg::Image::ConstSharedPtr at(double stamp_s) const;

    /// The newest frame no later than @p stamp_s, or the oldest held when all are later.
    [[nodiscard]] sensor_msgs::msg::Image::ConstSharedPtr atOrBefore(double stamp_s) const;

    [[nodiscard]] std::size_t size() const { return frames_.size(); }

    /// A header stamp in seconds.
    [[nodiscard]] static double stampSeconds(const std_msgs::msg::Header& header);

private:
    double                                              history_s_;
    double                                              tolerance_s_;
    std::size_t                                         max_frames_;
    double                                              min_spacing_s_;
    std::deque<sensor_msgs::msg::Image::ConstSharedPtr> frames_;
};

}  // namespace canopy_perception

#endif  // CANOPY_PERCEPTION__DEPTH_HISTORY_HPP_
