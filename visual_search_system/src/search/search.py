import numpy as np

class KDTreeSearch:
    """
    Performs nearest neighbor search on a KD-tree.

    Attributes:
        tree (KDTree): The KD-tree to search in.
        distance_func (callable): The function to use for distance calculation.
    """

    def __init__(self, tree, distance_func=None):
        """
        Initialize a KDTreeSearch.

        Args:
            tree (KDTree): The KD-tree to search in.
            distance_func (callable, optional): The function to use for distance calculation. Defaults to Euclidean distance.
        """
        self.tree = tree
        self.distance_func = distance_func if distance_func is not None else Measure.euclidean

    def nearest_neighbors(self, root, point, k, depth=0, neighbors=None):
        """
        Recursively find the nearest neighbors in the KD-tree.

        Args:
            root (KDTreeNode): The current root node.
            point (list or tuple): The target point.
            k (int): The number of nearest neighbors to find.
            depth (int, optional): The current depth in the tree. Defaults to 0.
            neighbors (list, optional): The list to store the nearest neighbors. Defaults to None.

        Returns:
            list: The nearest neighbors.
        """
        if root is None:
            return neighbors

        if neighbors is None:
            neighbors = []

        cd = depth % self.tree.k
        distance = self.distance_func(point, root.point)

        if len(neighbors) < k:
            neighbors.append((distance, root))
            neighbors.sort(reverse=True, key=lambda x: x[0])
        elif distance < neighbors[0][0]:
            neighbors[0] = (distance, root)
            neighbors.sort(reverse=True, key=lambda x: x[0])

        if point[cd] < root.point[cd]:
            neighbors = self.nearest_neighbors(root.left, point, k, depth + 1, neighbors)
            if len(neighbors) < k or abs(point[cd] - root.point[cd]) < neighbors[0][0]:
                neighbors = self.nearest_neighbors(root.right, point, k, depth + 1, neighbors)
        else:
            neighbors = self.nearest_neighbors(root.right, point, k, depth + 1, neighbors)
            if len(neighbors) < k or abs(point[cd] - root.point[cd]) < neighbors[0][0]:
                neighbors = self.nearest_neighbors(root.left, point, k, depth + 1, neighbors)

        return neighbors

    def find_nearest_neighbors(self, point, k):
        """
        Find the nearest neighbors to a given point.

        Args:
            point (list or tuple): The target point.
            k (int): The number of nearest neighbors to find.

        Returns:
            list: The nearest neighbors.
        """
        neighbors = self.nearest_neighbors(self.tree.root, point, k)
        if neighbors is None:
            return []
        return [(node.point, node.metadata) for _, node in sorted(neighbors, key=lambda x: x[0])]


class Measure:
    """
    Provides various distance metrics for nearest neighbor search.

    Methods:
        euclidean(point1, point2): Calculates the Euclidean distance between two points.
        cosine_similarity(point1, point2): Calculates the Cosine Similarity between two points.
        manhattan(point1, point2): Calculates the Manhattan distance between two points.
        minkowski(point1, point2, p): Calculates the Minkowski distance between two points.
    """

    @staticmethod
    def euclidean(point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (list or tuple): The first point.
            point2 (list or tuple): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    def cosine_similarity(point1, point2):
        """
        Calculate the Cosine Similarity between two points.

        Args:
            point1 (list or tuple): The first point.
            point2 (list or tuple): The second point.

        Returns:
            float: The Cosine Similarity between the two points.
        """
        p1 = np.array(point1)
        p2 = np.array(point2)
        return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

    @staticmethod
    def manhattan(point1, point2):
        """
        Calculate the Manhattan distance between two points.

        Args:
            point1 (list or tuple): The first point.
            point2 (list or tuple): The second point.

        Returns:
            float: The Manhattan distance between the two points.
        """
        return np.sum(np.abs(np.array(point1) - np.array(point2)))

    @staticmethod
    def minkowski(point1, point2, p=2):
        """
        Calculate the Minkowski distance between two points.

        Args:
            point1 (list or tuple): The first point.
            point2 (list or tuple): The second point.
            p (int): The order of the Minkowski distance.

        Returns:
            float: The Minkowski distance between the two points.
        """
        return np.sum(np.abs(np.array(point1) - np.array(point2)) ** p) ** (1 / p)


if __name__ == "__main__":
    from indexing import KDTree
    points = [
        [2, 3],
        [5, 4],
        [9, 6],
        [4, 7],
        [8, 1],
        [7, 2]
    ]

    metadata_list = [
        {"image_id": 1, "value": "Alice"},
        {"image_id": 2, "value": "Bob"},
        {"image_id": 3, "value": "Charlie"},
        {"image_id": 4, "value": "Daniel"},
        {"image_id": 5, "value": "Eve"},
        {"image_id": 6, "value": "Frank"}
    ]

    kd_tree = KDTree(k=2, points=points, metadata_list=points)
    kd_tree.print_tree()

    print("KD-Tree after insertion:")
    kd_tree.print_tree()

    # Using Euclidean distance
    search_euclidean = KDTreeSearch(kd_tree, Measure.euclidean)
    query_point = [10, 10]
    k = 2
    nearest_neighbors = search_euclidean.find_nearest_neighbors(query_point, k)
    print(f"\nTop {k} nearest neighbors to {query_point} using Euclidean distance:")
    for neighbor in nearest_neighbors:
        print(neighbor)

    # Using Manhattan distance
    search_manhattan = KDTreeSearch(kd_tree, Measure.manhattan)
    nearest_neighbors = search_manhattan.find_nearest_neighbors(query_point, k)
    print(f"\nTop {k} nearest neighbors to {query_point} using Manhattan distance:")
    for neighbor in nearest_neighbors:
        print(neighbor)

    # Using Cosine similarity (Note: For nearest neighbors, higher similarity is better, so we need to invert the metric)
    search_cosine = KDTreeSearch(kd_tree, lambda p1, p2: 1 - Measure.cosine_similarity(p1, p2))
    nearest_neighbors = search_cosine.find_nearest_neighbors(query_point, k)
    print(f"\nTop {k} nearest neighbors to {query_point} using Cosine similarity:")
    for neighbor in nearest_neighbors:
        print(neighbor)

    # Using Minkowski distance
    p = 5
    search_minkowski = KDTreeSearch(kd_tree, lambda p1, p2: Measure.minkowski(p1, p2, p))
    nearest_neighbors = search_minkowski.find_nearest_neighbors(query_point, k)
    print(f"\nTop {k} nearest neighbors to {query_point} using Minkowski distance:")
    for neighbor in nearest_neighbors:
        print(neighbor)