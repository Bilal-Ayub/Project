# -*- coding: utf-8 -*-

"""A Python implementation of a kd-tree

This package provides a simple implementation of a kd-tree in Python.
https://en.wikipedia.org/wiki/K-d_tree
"""

from __future__ import print_function

import heapq
import itertools
import operator
import math
from collections import deque


def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError(
                "All Points in the point_list must have the same dimensionality"
            )
    return dimensions


class Node(object):
    """A Node in a kd-tree

    A tree is represented by its root node, and every node represents
    its subtree"""

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        """Returns True if a Node has no subnodes"""
        return (not self.data) or (all(not bool(c) for c, _ in self.children))

    def preorder(self):
        """iterator for nodes: root, left, right"""
        if not self:
            return
        yield self
        if self.left:
            for x in self.left.preorder():
                yield x
        if self.right:
            for x in self.right.preorder():
                yield x

    def inorder(self):
        """iterator for nodes: left, root, right"""
        if not self:
            return
        if self.left:
            for x in self.left.inorder():
                yield x
        yield self
        if self.right:
            for x in self.right.inorder():
                yield x

    def postorder(self):
        """iterator for nodes: left, right, root"""
        if not self:
            return
        if self.left:
            for x in self.left.postorder():
                yield x
        if self.right:
            for x in self.right.postorder():
                yield x
        yield self

    def children(self):
        """Returns an iterator for non-empty children (node, pos)"""
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    def set_child(self, index, child):
        """Sets one of the node's children (0=left, 1=right)"""
        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        """Returns height of the (sub)tree, without counting empty leaves"""
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, _ in self.children])

    def get_child_pos(self, child):
        """Returns the position of the given child"""
        for c, pos in self.children:
            if child == c:
                return pos

    def __repr__(self):
        return "<%(cls)s - %(data)s>" % dict(
            cls=self.__class__.__name__, data=repr(self.data)
        )

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


class KDNode(Node):
    """A Node that contains kd-tree specific data and methods"""

    def __init__(
        self,
        data=None,
        left=None,
        right=None,
        axis=None,
        sel_axis=None,
        dimensions=None,
    ):
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    def add(self, point):
        """Adds a point to the tree or its appropriate subtree"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "add requires node %s to have axis and sel_axis" % repr(self)
            )
        current = self
        while True:
            check_dimensionality([point], dimensions=current.dimensions)
            if current.data is None:
                current.data = point
                return current
            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.create_subnode(point)
                    return current.left
                current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                current = current.right

    def create_subnode(self, data):
        """Creates a subnode for the current node"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "create_subnode requires node %s to have axis and sel_axis" % repr(self)
            )
        return self.__class__(
            data,
            axis=self.sel_axis(self.axis),
            sel_axis=self.sel_axis,
            dimensions=self.dimensions,
        )

    def find_replacement(self):
        """Finds replacement node for deletion"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "find_replacement requires node %s to have axis and sel_axis"
                % repr(self)
            )
        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)
        return child, parent if parent is not None else self

    def should_remove(self, point, node):
        if not self.data == point:
            return False
        return (node is None) or (node is self)

    def remove(self, point, node=None):
        """Removes the node with the given point from the tree"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "remove requires node %s to have axis and sel_axis" % repr(self)
            )
        if not self:
            return
        if self.should_remove(point, node):
            return self._remove(point)
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)
        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)
        if point[self.axis] <= self.data[self.axis] and self.left:
            self.left = self.left.remove(point, node)
        if point[self.axis] >= self.data[self.axis] and self.right:
            self.right = self.right.remove(point, node)
        return self

    def _remove(self, point):
        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self
        # deleting internal node: find replacement
        root, max_p = self.find_replacement()
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left = tmp_l if tmp_l is not root else self
        root.right = tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)
        else:
            root.remove(point, self)
        return root

    def is_balanced(self):
        """Returns True if the (sub)tree is balanced"""
        left_h = self.left.height() if self.left else 0
        right_h = self.right.height() if self.right else 0
        if abs(left_h - right_h) > 1:
            return False
        return all(c.is_balanced for c, _ in self.children)

    def rebalance(self):
        """Returns the root of the rebalanced tree"""
        return create([x.data for x in self.inorder()])

    def axis_dist(self, point, axis):
        """Squared axis distance between node and point"""
        return (self.data[axis] - point[axis]) ** 2

    def dist(self, point):
        """Squared distance between node and point"""
        return sum(self.axis_dist(point, i) for i in range(self.dimensions))

    def search_knn(self, point, k, dist=None):
        """Return the k nearest neighbors of point and their distances"""
        if k < 1:
            raise ValueError("k must be greater than 0.")
        get_dist = (
            (lambda n: n.dist(point))
            if dist is None
            else (lambda n: dist(n.data, point))
        )
        results = []
        self._search_node(point, k, results, get_dist, itertools.count())
        return [(node, -d) for d, _, node in sorted(results, reverse=True)]

    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return
        nodeDist = get_dist(self)
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        split = self.data[self.axis]
        pd = point[self.axis] - split
        pd2 = pd * pd
        if point[self.axis] < split and self.left:
            self.left._search_node(point, k, results, get_dist, counter)
        elif self.right:
            self.right._search_node(point, k, results, get_dist, counter)
        if -pd2 > results[0][0] or len(results) < k:
            if point[self.axis] < split and self.right:
                self.right._search_node(point, k, results, get_dist, counter)
            elif self.left:
                self.left._search_node(point, k, results, get_dist, counter)

    def search_nn(self, point, dist=None):
        """Search the nearest node of the given point"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "search_nn requires node %s to have axis and sel_axis" % repr(self)
            )
        return next(iter(self.search_knn(point, 1, dist)), None)

    def _search_nn_dist(self, point, dist_val, results, get_d):
        if not self:
            return
        d = get_d(self)
        if d < dist_val:
            results.append(self.data)
        split = self.data[self.axis]
        if point[self.axis] <= split + dist_val and self.left:
            self.left._search_nn_dist(point, dist_val, results, get_d)
        if point[self.axis] >= split - dist_val and self.right:
            self.right._search_nn_dist(point, dist_val, results, get_d)

    def search_nn_dist(self, point, distance, best=None):
        """Search nodes within given distance"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "search_nn_dist requires node %s to have axis and sel_axis" % repr(self)
            )
        results = []
        get_d = lambda n: n.dist(point)
        self._search_nn_dist(point, distance, results, get_d)
        return results

    def is_valid(self):
        """Checks recursively if the tree is valid"""
        if None in (self.axis, self.sel_axis):
            raise ValueError(
                "is_valid requires node %s to have axis and sel_axis" % repr(self)
            )
        if not self:
            return True
        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False
        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False
        return all(c.is_valid() for c, _ in self.children) or self.is_leaf


def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """Creates a kd-tree from a list of points"""
    if not point_list and not dimensions:
        raise ValueError("either point_list or dimensions must be provided")
    if point_list:
        dimensions = check_dimensionality(point_list, dimensions)
    sel_axis = sel_axis or (lambda prev: (prev + 1) % dimensions)
    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)
    pts = list(point_list)
    pts.sort(key=lambda p: p[axis])
    mid = len(pts) // 2
    loc = pts[mid]
    left = create(pts[:mid], dimensions, sel_axis(axis))
    right = create(pts[mid + 1 :], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def level_order(tree, include_all=False):
    """Returns an iterator over the tree in level-order"""
    q = deque([tree])
    while q:
        node = q.popleft()
        yield node
        if include_all or node.left:
            q.append(node.left or node.__class__())
        if include_all or node.right:
            q.append(node.right or node.__class__())


def visualize(tree, max_level=100, node_width=10, left_padding=5):
    """Prints the tree to stdout"""
    height = min(max_level, tree.height() - 1)
    max_width = 2**height
    per = 1
    in_lvl = 0
    lvl = 0
    for node in level_order(tree, include_all=True):
        if in_lvl == 0:
            print()
            print()
            print(" " * left_padding, end=" ")
        width = int(max_width * node_width / per)
        s = (str(node.data) if node else "").center(width)
        print(s, end=" ")
        in_lvl += 1
        if in_lvl == per:
            in_lvl = 0
            per *= 2
            lvl += 1
        if lvl > height:
            break
    print()
    print()


# Recreate properties removed via decorators
Node.is_leaf = property(Node.is_leaf)
Node.children = property(Node.children)
KDNode.is_balanced = property(KDNode.is_balanced)


# main

p1 = (2, 3, 4)
p2 = (4, 5, 6)
p3 = (5, 3, 2)

# each sub-tree represented by a root node
tree = create([p1, p2, p3])
visualize(tree)
tree.add((5, 4, 3))
visualize(tree)
tree.remove((5, 4, 3))
visualize(tree)

print(list(tree.inorder()))
print(list(level_order(tree)))

print(tree.search_nn((1, 2, 3)))
tree.add((10, 2, 1))
visualize(tree)

subtree = tree.right
tree.right = None
visualize(tree)
visualize(subtree)

tree.right = subtree
visualize(tree)

print(tree.is_balanced)

tree.add((6, 1, 5))
visualize(tree)
print(tree.is_balanced)

visualize(tree)

tree = tree.rebalance()
print(tree.is_balanced)

visualize(tree)
