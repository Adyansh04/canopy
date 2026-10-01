# Something that happened to an object of the world model. Missing means looked at and not found:
# the depth saw through where it stood. A missed detection alone never makes an object missing.

builtin_interfaces/Time stamp

# "O12".
string id
# Its label then.
string label

# appeared, moved, missing, seen_again, removed or merged.
string kind

# Where it stood then: its footprint's centre, map frame. z is unused.
geometry_msgs/Point position
# The room that point is in now; empty when it is in none.
string room_id

# appeared: an object of its label that went missing before it appeared, which it may be after a
# move. merged: the object it turned out to be part of, which goes on from here.
string other_id

# removed: why, such as "clean-up: floor".
string detail
