import matplotlib.pyplot as plt
import math
import numpy as np

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counterclockwise

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    elif o1 == 0 and on_segment(p1, p2, q1):
        return True
    elif o2 == 0 and on_segment(p1, q2, q1):
        return True
    elif o3 == 0 and on_segment(p2, p1, q2):
        return True
    elif o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def check_intersection(points, new_point):

    # Check intersection between the last segment and all other segments
    for i in range(len(points) - 2):
        if do_intersect(points[i], points[i + 1], points[-1], new_point):
            return True
    return False


def draw_line_segments(line_segments, ax):
    for i in range(len(line_segments) - 1):
        p1, p2 = line_segments[i], line_segments[i + 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='.', color='blue')

    # Set equal aspect ratio for X and Y axes
    ax.set_aspect('equal', adjustable='box')



def clockwise_angle(p1, p2, p3):
    va = np.array([p2[0] - p1[0], p2[1] - p1[1], 0])
    vb = np.array([p3[0] - p2[0], p3[1] - p2[1], 0])
    vn = np.array([0, 0, 1])

    cross_product = np.cross(va, vb)
    dot_product = np.dot(va, vb)
    angle = np.arctan2(np.dot(cross_product, vn), dot_product)

    return angle


def add_most_clockwise_point(points, rise_run):
    dx, dy = rise_run

    if len(points) == 0:
        points.append((0, 0))
    if len(points) == 1:
        points.append((dx, dy))
        return True

    last_point = points[-1]

    # Generate all possible points with the given rise/run
    candidate_points = [
        (last_point[0] + dx, last_point[1] + dy),
        (last_point[0] - dx, last_point[1] + dy),
        (last_point[0] + dx, last_point[1] - dy),
        (last_point[0] - dx, last_point[1] - dy),
        (last_point[0] + dy, last_point[1] + dx),
        (last_point[0] - dy, last_point[1] + dx),
        (last_point[0] + dy, last_point[1] - dx),
        (last_point[0] - dy, last_point[1] - dx),
    ]

    # Sort candidate points by their clockwise angle with the x-axis
    # Sort candidate points by their clockwise angle with the last line segment
    candidate_points.sort(
        key=lambda p: clockwise_angle(points[-2], last_point, p)
    )

    #random.shuffle(candidate_points)

    # Find the most clockwise point that doesn't intersect existing line segments
    for candidate_point in candidate_points:
        if not check_intersection(points, candidate_point):
            points.append(candidate_point)
            return True

    print("col")
    return False

def plot_sequence(seq, ax, maxiter=100):
    points = []
    for i in range(maxiter):
        if not add_most_clockwise_point(points, seq[i % len(seq)]):
            break
    
    return draw_line_segments(points, ax)

if __name__ == "__main__":

    fig, ax = plt.subplots(2,2)

    plot_sequence([(1,0),(1,1)], ax[0, 0])
    plot_sequence([(1,1),(1,2)], ax[0, 1])
    plot_sequence([(1,1),(1,3)], ax[1, 1])
    plot_sequence([(1,1),(1,4)], ax[1, 0])

    plt.show()
