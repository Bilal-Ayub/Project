import numpy as np
import math


# Class for vector
# class Vector:
#     def __init__(self, x, y, z): #values are initialized (x,y,z) dimensions
#         self.x = x
#         self.y = y
#         self.z = z

#     def __sub__(self, other):  # vector subtraction
#         return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

#     def __add__(self, other):  # vector addition -> returns new vector after this operation
#         return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

#     def dot(self, other): #scalar dot product of vector
#         return self.x * other.x + self.y * other.y + self.z * other.z

#     def length(self): # uses dot product to find the magnitude of vector
#         return np.sqrt(self.dot(self))

#     def normalize(self): #returns unit vector (divides each component by its magnitude: length here)
#         length = self.length()
#         return Vector(self.x / length, self.y / length, self.z / length)

""" Define vector using NumPy array. The operation carries our further mathematical operations and returns
NumPy arrays that contains the x,y,z coords of the shape we are rendering"""


def vector(x, y, z):
    """Creates a 3-component vector as a tuple of floats."""
    return (float(x), float(y), float(z))


def vector_addition(v1, v2):
    """Returns the addition of two vectors."""
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


def vector_subtraction(v1, v2):
    """Returns the subtraction of two vectors."""
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def dot_product(v1, v2):
    """Returns the dot product of two vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def length_of_vector(v):
    """Returns the magnitude of a vector."""
    return math.sqrt(dot_product(v, v))


def normalize(v):
    """Returns the unit vector in the direction of v."""
    norm = length_of_vector(v)
    if norm == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


class Ray:
    """Ray with origin and direction (unit vector)."""

    def __init__(self, origin, direction):
        self.origin = origin  # tuple of floats
        self.direction = normalize(direction)


class Sphere:
    """Sphere defined by center (vector) and radius."""

    def __init__(self, center, radius):
        self.center = center  # tuple of floats
        self.radius = radius

    def intersect(self, ray):
        """Ray-sphere intersection. Returns smallest positive t or None."""
        oc = vector_subtraction(ray.origin, self.center)
        a = dot_product(ray.direction, ray.direction)
        b = 2.0 * dot_product(oc, ray.direction)
        c = dot_product(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        return None


class KdNode:  # part of kdtree
    def __init__(self, left, right, objects):
        self.left = left
        self.right = right
        self.objects = objects  # spheres


# Build a simple kd-tree (for demonstration purposes)
def build_kd_tree(objects, depth=0):
    if not objects:
        return None

    # Choose axis based on depth
    axis = depth % 3

    # Sort objects along the chosen axis
    # objects.sort(key=lambda obj: obj.center.x if axis == 0 else obj.center.y if axis == 1 else obj.center.z)
    def sort_key(obj):
        if axis == 0:
            return obj.center[0]
        elif axis == 1:
            return obj.center[1]
        else:
            return obj.center[2]

    objects.sort(key=sort_key)

    # Find the median
    median = len(objects) // 2

    # Recursively build the tree
    return KdNode(
        left=build_kd_tree(objects[:median], depth + 1),
        right=build_kd_tree(objects[median + 1 :], depth + 1),
        objects=[objects[median]],
    )


# Traverse the kd-tree to find intersections
def intersect_kd_tree(ray, node):  # find the smallest intersection
    if node is None:
        return None

    # Check intersection with the current node's objects
    hit = None
    for obj in node.objects:
        t = obj.intersect(ray)
        if t is not None and (hit is None or t < hit):
            hit = t

    # Traverse the left and right subtrees
    hit_left = intersect_kd_tree(ray, node.left)
    hit_right = intersect_kd_tree(ray, node.right)

    # Find the closest hit
    if hit_left is not None and (hit is None or hit_left < hit):
        hit = hit_left
    if hit_right is not None and (hit is None or hit_right < hit):
        hit = hit_right

    return hit


def test_case_1():
    sphere = Sphere(vector(0, 0, 0), 0.5)
    ray = Ray(vector(0, 0, -1), vector(0, 0, 1))
    kd_tree = build_kd_tree([sphere])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is not None, "Test Case 1 Failed: No intersection found"
    print(f"Test Case 1 Passed: Intersection at t = {t}")


def test_case_2():
    sphere = Sphere(vector(0, 0, 0), 0.5)
    ray = Ray(vector(2, 2, -1), vector(0, 0, 1))
    kd_tree = build_kd_tree([sphere])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is None, "Test Case 2 Failed: Unexpected intersection found"
    print("Test Case 2 Passed: No intersection (ray misses the sphere)")


def test_case_3():
    sphere1 = Sphere(vector(0, 0, 1), 0.5)
    sphere2 = Sphere(vector(0, 0, 3), 0.5)
    ray = Ray(vector(0, 0, -1), vector(0, 0, 1))
    kd_tree = build_kd_tree([sphere1, sphere2])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is not None and t < 2.0, (
        "Test Case 3 Failed: Intersection with the farther sphere"
    )
    print(f"Test Case 3 Passed: Intersection at t = {t} (closer sphere)")


def test_case_4():
    sphere1 = Sphere(vector(0, 0, 1), 0.5)
    sphere2 = Sphere(vector(0, 0.6, 3), 0.5)
    ray = Ray(vector(0, 1, -1), vector(0, 0, 1))
    kd_tree = build_kd_tree([sphere1, sphere2])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is not None and t > 2.0, (
        "Test Case 4 Failed: Intersection with the closer sphere"
    )
    print(f"Test Case 4 Passed: Intersection at t = {t} (farther sphere)")


def test_case_5():
    ray = Ray(vector(0, 0, -1), vector(0, 0, 1))
    kd_tree = build_kd_tree([])
    t = intersect_kd_tree(ray, kd_tree)
    assert t is None, "Test Case 5 Failed: Unexpected intersection found"
    print("Test Case 5 Passed: No intersection (empty kd-tree)")


if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
