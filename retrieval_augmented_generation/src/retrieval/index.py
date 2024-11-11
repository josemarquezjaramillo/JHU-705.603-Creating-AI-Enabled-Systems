class Node:
    """A node in a KDTree.

    Attributes:
        point (list): The point represented by this node.
        metadata (dict): Metadata associated with the point.
        left (Node, optional): Left child node. Defaults to None.
        right (Node, optional): Right child node. Defaults to None.
    """

    def __init__(self, point, metadata=None, left=None, right=None):
        """Initialize a Node.

        Args:
            point (list): The point represented by this node.
            metadata (dict, optional): Metadata associated with the point. Defaults to None.
            left (Node, optional): Left child node. Defaults to None.
            right (Node, optional): Right child node. Defaults to None.
        """
        self.point = point
        self.metadata = metadata if metadata is not None else {}
        self.left = left
        self.right = right


class KDTree:
    """A KDTree implementation.

    Attributes:
        k (int): Number of dimensions.
        root (Node): Root node of the KDTree.
    """

    def __init__(self, k, points, metadata_list=None):
        """Initialize the KDTree.

        Args:
            points (list of list): List of points to build the KDTree.
            metadata_list (list of dict, optional): List of metadata dictionaries for each point. Defaults to None.
        """
        self.k = k
        self.root = self._build_tree(points, metadata_list, depth=0)

    def _build_tree(self, points, metadata_list, depth):
        """Recursively build the KDTree.

        Args:
            points (list of list): List of points.
            metadata_list (list of dict, optional): List of metadata dictionaries for each point. Defaults to None.
            depth (int): Current depth in the tree.

        Returns:
            Node: The root node of the KDTree.
        """
        if not points:
            return None

        axis = depth % self.k
        points_with_metadata = list(zip(points, metadata_list)) if metadata_list else [(p, {}) for p in points]
        points_with_metadata.sort(key=lambda x: x[0][axis])
        median = len(points_with_metadata) // 2

        point, metadata = points_with_metadata[median]
        return Node(
            point=point,
            metadata=metadata,
            left=self._build_tree([p for p, _ in points_with_metadata[:median]],
                                  [m for _, m in points_with_metadata[:median]], depth + 1),
            right=self._build_tree([p for p, _ in points_with_metadata[median + 1:]],
                                   [m for _, m in points_with_metadata[median + 1:]], depth + 1)
        )

    def _insert(self, root, point, metadata, depth):
        """Recursively insert a point into the KDTree.

        Args:
            root (Node): The current root node.
            point (list): The point to insert.
            metadata (dict): Metadata associated with the point.
            depth (int): Current depth in the tree.

        Returns:
            Node: The updated root node.
        """
        if root is None:
            return Node(point, metadata)

        axis = depth % self.k

        if point[axis] < root.point[axis]:
            root.left = self._insert(root.left, point, metadata, depth + 1)
        else:
            root.right = self._insert(root.right, point, metadata, depth + 1)

        return root

    def insert(self, point, metadata=None):
        """Insert a point into the KDTree.

        Args:
            point (list): The point to insert.
            metadata (dict, optional): Metadata associated with the point. Defaults to None.
        """
        metadata = metadata if metadata is not None else {}
        self.root = self._insert(self.root, point, metadata, depth=0)

    def _find_min(self, root, d, depth):
        """Find the node with the minimum value in a specific dimension.

        Args:
            root (Node): The current root node.
            d (int): The dimension to find the minimum in.
            depth (int): Current depth in the tree.

        Returns:
            Node: The node with the minimum value in the specified dimension.
        """
        if root is None:
            return None

        axis = depth % self.k

        if axis == d:
            if root.left is None:
                return root
            return self._find_min(root.left, d, depth + 1)

        return min(root,
                   self._find_min(root.left, d, depth + 1),
                   self._find_min(root.right, d, depth + 1),
                   key=lambda x: x.point[d] if x else float('inf'))

    def _delete_node(self, root, point, depth):
        """Recursively delete a point from the KDTree.

        Args:
            root (Node): The current root node.
            point (list): The point to delete.
            depth (int): Current depth in the tree.

        Returns:
            Node: The updated root node.
        """
        if root is None:
            return None

        axis = depth % self.k

        if root.point == point:
            if root.right is not None:
                min_node = self._find_min(root.right, axis, depth + 1)
                root.point = min_node.point
                root.metadata = min_node.metadata
                root.right = self._delete_node(root.right, min_node.point, depth + 1)
            elif root.left is not None:
                min_node = self._find_min(root.left, axis, depth + 1)
                root.point = min_node.point
                root.metadata = min_node.metadata
                root.right = self._delete_node(root.left, min_node.point, depth + 1)
                root.left = None
            else:
                return None
        elif point[axis] < root.point[axis]:
            root.left = self._delete_node(root.left, point, depth + 1)
        else:
            root.right = self._delete_node(root.right, point, depth + 1)

        return root

    def remove(self, point):
        """Remove a point from the KDTree.

        Args:
            point (list): The point to remove.
        """
        self.root = self._delete_node(self.root, point, depth=0)

    def _print_tree(self, node, depth=0, prefix="Root: ", by=None):
        """Recursively print the KDTree.

        Args:
            node (Node): The current node.
            depth (int): Current depth in the tree.
            prefix (str): Prefix for printing the current node.
        """
        if not node:
            return

        if by:
            print(" " * depth * 2 + prefix + f" {by}: " + str(node.metadata[by]))
        else:
            print(" " * depth * 2 + prefix + str(node.point) + " Metadata: " + str(node.metadata))
        if node.left:
            self._print_tree(node.left, depth + 1, prefix="L--- ", by=by)
        if node.right:
            self._print_tree(node.right, depth + 1, prefix="R--- ", by=by)

    def print_tree(self, by=None):
        """Print the KDTree."""
        self._print_tree(self.root, by=by)


if __name__ == "__main__":
    points = [
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ]

    metadata_list = [
        {"image_id": 1, "name": "Alice"},
        {"image_id": 2, "name": "Bob"},
        {"image_id": 3, "name": "Charlie"},
        {"image_id": 4, "name": "Daniel"},
        {"image_id": 5, "name": "Eve"},
        {"image_id": 6, "name": "Frank"}
    ]

    kd_tree = KDTree(k=2, points=points, metadata_list=metadata_list)
    kd_tree.print_tree(by="name")

    # Adding a point with metadata
    print("\nAfter adding [3, 6] with metadata {'image_id': 7, 'name': 'Grace'}:")
    kd_tree.insert([3, 6], metadata={"image_id": 7, "name": "Grace"})
    kd_tree.print_tree(by="name")

    # Removing a point
    print("\nAfter removing [5, 4]:")
    kd_tree.remove([5, 4])
    kd_tree.print_tree(by="name")
