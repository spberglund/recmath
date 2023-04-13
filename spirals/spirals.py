import matplotlib.pyplot as plt
import numpy as np
import colorsys

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


def draw_line_segments(points, ax, title):

    if points:
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            clr = tuple(int(c * 255) for c in colorsys.hls_to_rgb(1.0 - (i/len(points)), 0.5, 0.8))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='', color=f'#{clr[0]:02x}{clr[1]:02x}{clr[2]:02x}')

    ax.set_title(title)
    ax.set_axis_off()
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

    return False

def exec_sequence(seq, maxiter=500):
    points = []
    for i in range(maxiter):
        if not add_most_clockwise_point(points, seq[i % len(seq)]):
            break

    return points


def iter_riseruns(n):

    rr = [0,0]

    i = 0

    while n is None or i < n:
        i += 1
        if rr[0] >= rr[1]:
            rr[0] = 0
            rr[1] += 1
        else:
            rr[0] += 1

        yield tuple(rr)

def is_base_seq(seq):
    gcds = [np.gcd(a, b) for a, b in seq]
    gcdall = gcds[0]

    for n in gcds[1:]:
        gcdall = np.gcd(gcdall, n)

    return gcdall == 1

def iter_seqs(dims):
    rr_iters = [iter_riseruns(None) for i in range(dims)]
    cur_seq = [(0,1)]*dims

    d = 0

    while True:
        cur_seq[d] = next(rr_iters[d])

        if is_base_seq(cur_seq):
            yield tuple(cur_seq)

        d += 1
        if d >= dims:
            d = 0


if __name__ == "__main__":

    maxiter = 500

    #Draw one
    # seq = ((1,2), (2,3))

    # points = exec_sequence(seq, maxiter)

    # numsegs = len(points)-1
    # title = f"{' → '.join(f'{a}x{b}' for a, b in seq)}{'=' if numsegs < maxiter else '≥'}{numsegs}"

    # print(title)

    # fig, ax = plt.subplots()
    # draw_line_segments(points, ax, title)
    # plt.show()
    
    # Draw multiple
    dimx, dimy = 10, 10
    fig, ax = plt.subplots(dimx, dimy)

    for i, seq in enumerate(iter_seqs(2)):

        if i >= dimx * dimy:
            break

        x, y = i // dimy, i % dimx

        print(seq)
        if not is_base_seq(seq):
            print("DUPLICATE")
            draw_line_segments(None, ax[x, y], None)
            continue

        points = exec_sequence(seq, maxiter)

        numsegs = len(points)-1
        title = f"{' → '.join(f'{a}x{b}' for a, b in seq)}{'=' if numsegs < maxiter else '≥'}{numsegs}"

        print(title)

        draw_line_segments(points, ax[x, y], title)

    plt.show()
