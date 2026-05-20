std_msgs/Header header
# header.stamp is the acquisition time of channel 0 sample 0 in ROS time.
# Prefer SDK/sample-clock timestamps anchored to ROS time; arrival-time stamps
# should be an explicit fallback in publishers.
string source_id
float32 sampling_rate_hz
# Ordered channel labels. The order defines the channel axis and must remain
# stable across a recording/session/model unless a publisher applies a named
# remap before publishing.
string[] channel_labels
# Raw EEG samples in microvolts, channel-major:
# [ch0_sample0, ch0_sample1, ..., ch1_sample0, ch1_sample1, ...].
float32[] samples
