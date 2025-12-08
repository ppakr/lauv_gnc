import matplotlib.pyplot as plt
import numpy as np
import math


def generate_smooth_lawn_mower(
    start_x, start_y, leg_length, spacing, num_legs, turn_radius
):
    """
    Generates a lawn mower path with correct Dubins U-turns.
    """
    waypoints = []

    # Adjust radius if spacing is too tight
    if spacing < 2 * turn_radius:
        print(
            f"Warning: Spacing ({spacing}m) is too tight for Turn Radius ({turn_radius}m)."
        )
        turn_radius = spacing / 2.0

    # Helper to generate arc points
    def add_arc(center_x, center_y, theta_start, theta_end, radius, num_points=15):
        # Generate angles.
        # For Top Turn (CW): pi -> 0
        # For Bottom Turn (CCW): pi -> 2pi
        angles = np.linspace(theta_start, theta_end, num_points)

        for ang in angles:
            wx = center_x + radius * math.cos(ang)
            wy = center_y + radius * math.sin(ang)
            waypoints.append((wx, wy))

    # --- Generation Loop ---
    for i in range(num_legs):
        # Calculate X for this leg
        current_x = start_x + (i * spacing)

        # Determine Direction
        is_even = i % 2 == 0  # Even = UP, Odd = DOWN

        if is_even:
            # --- GOING UP (South -> North) ---
            # 1. Start of leg
            if i == 0:
                waypoints.append((current_x, start_y))

            # 2. Straight line to Top
            waypoints.append((current_x, start_y + leg_length))

            # 3. U-Turn at Top (Right Turn / Clockwise)
            if i < num_legs - 1:
                center_x = current_x + (spacing / 2.0)
                center_y = start_y + leg_length

                # Arc: Starts at West point (pi), goes Clockwise to East point (0)
                # Curve is ABOVE the line (sin is positive)
                add_arc(center_x, center_y, math.pi, 0, spacing / 2.0)

        else:
            # --- GOING DOWN (North -> South) ---
            # 1. Straight line to Bottom
            # (We are already at the top from the previous turn)
            waypoints.append((current_x, start_y))

            # 2. U-Turn at Bottom (Left Turn / Counter-Clockwise)
            if i < num_legs - 1:
                center_x = current_x + (spacing / 2.0)
                center_y = start_y

                # Arc: Starts at West point (pi), goes Counter-Clockwise to East point (2pi)
                # Curve is BELOW the line (sin is negative)
                add_arc(center_x, center_y, math.pi, 2 * math.pi, spacing / 2.0)

    return waypoints


def plot_path(waypoints):
    x = [p[0] for p in waypoints]
    y = [p[1] for p in waypoints]

    plt.figure(figsize=(8, 10))
    plt.plot(x, y, marker=".", linestyle="-", linewidth=1.5, label="Trajectory")

    # Mark Start/End and Direction
    plt.plot(x[0], y[0], "go", markersize=10, label="Start")
    plt.plot(x[-1], y[-1], "rx", markersize=10, label="End")

    # Add arrows to verify direction
    arrow_idx = [len(waypoints) // 4, len(waypoints) // 2, 3 * len(waypoints) // 4]
    for idx in arrow_idx:
        if idx + 1 < len(x):
            plt.annotate(
                "",
                xy=(x[idx + 1], y[idx + 1]),
                xytext=(x[idx], y[idx]),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Dubins Lawn Mower (Spacing={SPACING}m, Leg Length={LEG_LENGTH}m)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.show()


def print_waypoints(waypoints):
    for i, (x, y) in enumerate(waypoints):
        print(f"Waypoint {i + 1}: X = {x:.2f}, Y = {y:.2f}")


if __name__ == "__main__":
    # --- Configuration ---
    START_X = 0.0
    START_Y = 0.0
    LEG_LENGTH = 50.0
    SPACING = 10.0  # Distance between survey lines
    NUM_LEGS = 5  # Total passes
    RADIUS = SPACING / 2.0  # Minimum turning radius constraint

    path = generate_smooth_lawn_mower(
        START_X, START_Y, LEG_LENGTH, SPACING, NUM_LEGS, RADIUS
    )

    print_waypoints(path)  # Uncomment to see all waypoint coordinates
    # Verify the points
    print(f"Generated {len(path)} waypoints.")
    plot_path(path)
