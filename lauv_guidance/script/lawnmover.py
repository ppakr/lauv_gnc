import matplotlib.pyplot as plt
import numpy as np
import math


def generate_smooth_lawn_mower(
    start_x, start_y, leg_length, spacing, num_legs, turn_radius
):
    """
    Generates a horizontal lawn mower path (X-axis legs) with Dubins U-turns.
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
        angles = np.linspace(theta_start, theta_end, num_points)
        for ang in angles:
            wx = center_x + radius * math.cos(ang)
            wy = center_y + radius * math.sin(ang)
            waypoints.append((wx, wy))

    # --- Generation Loop ---
    for i in range(num_legs):
        # Calculate Y for this leg (moves Up/North each leg)
        current_y = start_y + (i * spacing)

        # Determine Direction
        is_even = i % 2 == 0  # Even = Right/East, Odd = Left/West

        if is_even:
            # --- GOING RIGHT (West -> East) ---
            # 1. Start of leg
            if i == 0:
                waypoints.append((start_x, current_y))

            # 2. Straight line to Right End
            waypoints.append((start_x + leg_length, current_y))

            # 3. U-Turn at Right End (Left Turn / CCW)
            if i < num_legs - 1:
                center_x = start_x + leg_length
                center_y = current_y + (spacing / 2.0)

                # Arc: Starts at Bottom (-pi/2), goes CCW to Top (pi/2)
                # Bulges to the East (Positive X)
                add_arc(center_x, center_y, -math.pi / 2, math.pi / 2, spacing / 2.0)

        else:
            # --- GOING LEFT (East -> West) ---
            # 1. Straight line to Left End
            waypoints.append((start_x, current_y))

            # 2. U-Turn at Left End (Right Turn / CW)
            if i < num_legs - 1:
                center_x = start_x
                center_y = current_y + (spacing / 2.0)

                # Arc: Starts at Bottom (3pi/2), goes CW to Top (pi/2)
                # Bulges to the West (Negative X)
                add_arc(center_x, center_y, 3 * math.pi / 2, math.pi / 2, spacing / 2.0)

    return waypoints


def plot_path(waypoints):
    x = [p[0] for p in waypoints]
    y = [p[1] for p in waypoints]

    plt.figure(figsize=(10, 6))
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
    plt.title("Horizontal Dubins Lawn Mower")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # --- Configuration ---
    START_X = 0.0
    START_Y = 0.0
    LEG_LENGTH = 50.0  # Length along X
    SPACING = 10.0  # Step along Y
    NUM_LEGS = 5
    RADIUS = 5.0

    path = generate_smooth_lawn_mower(
        START_X, START_Y, LEG_LENGTH, SPACING, NUM_LEGS, RADIUS
    )

    print(f"Generated {len(path)} waypoints.")
    plot_path(path)
