"""Skeleton topology of the 22-joint HumanML3D motion.

Joint order follows HumanML3D / MoMask: 0 is the pelvis, 1-2 the hips, 3 the
lower spine, and so on. Each chain below is drawn as one polyline in the
rendered video, which is why the skeleton is stored as chains rather than as a
parent array.
"""

#: Chains of joint indices, drawn as polyline segments.
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],  # left leg
    [0, 3, 6, 9, 12, 15],  # spine and head
    [9, 14, 17, 19, 21],  # left arm
    [9, 13, 16, 18, 20],  # right arm
]

#: One colour per chain, kept in the same order as :data:`KINEMATIC_CHAIN`.
CHAIN_COLORS = ["red", "blue", "black", "darkred", "darkblue"]

JOINTS_NUM = 22

#: Axis meaning of the stored 3D joints.
AXIS_X = 0  # lateral
AXIS_Y = 1  # vertical, up
AXIS_Z = 2  # forward

#: The 2D tokenizer stores the frontal projection, i.e. (x, y).
PROJECTION_AXES_2D = (AXIS_X, AXIS_Y)
