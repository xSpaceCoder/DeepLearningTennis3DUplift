import numpy as np
# Tennis Court Dimensions (in meters, standard single court)
# Origin of the world coordinate system is at the center of the net, on the ground (z=0).
# The court extends along the x-axis, and the net lies on the y-axis.
TENNIS_COURT_LENGTH = 23.774  # baseline to baseline
TENNIS_COURT_WIDTH = 8.233  # sideline to sideline
TENNIS_COURT_DOUBLE_WIDTH = 10.97

# Net Dimensions
NET_TENNIS_WIDTH = TENNIS_COURT_WIDTH + (2 * 0.914)  # Length of the net
NET_TENNIS_HEIGHT = 1.07  # Height of the net at the center

# Derived Court and Net Positions
BASELINE_X_CLOSE = TENNIS_COURT_LENGTH / 2
BASELINE_X_FAR = -TENNIS_COURT_LENGTH / 2
SERVICELINE_X_CLOSE = 6.401
SERVICELINE_X_FAR = -6.401

SINGLE_SIDELIINE_Y_LEFT = -TENNIS_COURT_WIDTH / 2
SINGLE_SIDELIINE_Y_RIGHT = TENNIS_COURT_WIDTH / 2
DOUBLE_SIDELIINE_Y_LEFT = -TENNIS_COURT_DOUBLE_WIDTH / 2
DOUBLE_SIDELIINE_Y_RIGHT = TENNIS_COURT_DOUBLE_WIDTH / 2
NETPOST_Y_LEFT = -NET_TENNIS_WIDTH / 2
NETPOST_Y_RIGHT = NET_TENNIS_WIDTH / 2

# Tennis Court Points for visualization (origin at (0,0,0) = center of the net)
# These points define the keyoints of a single tennis court.
# X-axis is along the length, Y-axis is along the width, z is height
# Z=0 is the court surface.
court_points = np.array(
    [
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 0 close left
        [BASELINE_X_CLOSE, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 1 close right
        [0.0, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 2 center left
        [0.0, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 3 center right
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 4 far left
        [BASELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 5 far right
        [0.0, NETPOST_Y_LEFT, 0.0],  # 6 net left bottom
        [0.0, NETPOST_Y_RIGHT, 0.0],  # 7 net right bottom
        [0.0, 0.0, 0.0],  # 8 net center bottom
        [0.0, NETPOST_Y_LEFT, NET_TENNIS_HEIGHT],  # 9 net left top
        [0.0, NETPOST_Y_RIGHT, NET_TENNIS_HEIGHT],  # 10 net right top
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_LEFT,
            0.0,
        ],  # 11 service line close left
        [SERVICELINE_X_CLOSE, 0.0, 0.0],  # 12 service line close center
        [
            SERVICELINE_X_CLOSE,
            SINGLE_SIDELIINE_Y_RIGHT,
            0.0,
        ],  # 13 service line close right
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_LEFT, 0.0],  # 14 service line far left
        [SERVICELINE_X_FAR, 0.0, 0.0],  # 15 service line far center
        [SERVICELINE_X_FAR, SINGLE_SIDELIINE_Y_RIGHT, 0.0],  # 16 service line far right
        [0.0, 0.0, 0.914],  # 17 net center top
        [BASELINE_X_CLOSE, DOUBLE_SIDELIINE_Y_LEFT, 0.0],  # 18 double line close left
        [BASELINE_X_CLOSE, DOUBLE_SIDELIINE_Y_RIGHT, 0.0],  # 19 double line close right
        [BASELINE_X_FAR, DOUBLE_SIDELIINE_Y_LEFT, 0.0],  # 20 double line far left
        [BASELINE_X_FAR, DOUBLE_SIDELIINE_Y_RIGHT, 0.0],  # 21 double line far right
    ]
)

court_connections = [
    (0, 2),  # left side
    (2, 4),  # left side
    (1, 3),  # right side
    (3, 5),  # right side
    (0, 1),  # front side
    (4, 5),  # back side
    (6, 2),  # center line
    (2, 3),  # center line
    (3, 7),  # center line
    (6, 9),  # net
    (9, 17),  # net
    (17, 10),
    (10, 7),  # net
    (12, 8),  # center service line
    (8, 15),  # center service line
    (11, 12),  # close service line
    (12, 13),  # close service line
    (14, 15),  # far service line
    (15, 16),  # far service line
    (18, 19),
    (18, 20),
    (20, 21),
    (19, 21),
]