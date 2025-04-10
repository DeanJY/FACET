import cv2
import numpy as np


class PupilTracker:
    def __call__(self, event_frame_now, ellipse_last):
        """
        Get the new ellipse.

        Args:
            event_frame_now: numpy.ndarray, the current event frame.
            ellipse_last: tuple, (x, y, a, b, ang), the parameters of the last ellipse.
            x, y: int, the coordinates of the center of the ellipse.
            a, b: int, the major and minor axes of the ellipse, not half of the axes.
            ang: int, the angle of the ellipse.
        Returns:
            new_ellipse: tuple, the new ellipse.
        """
        self.event_frame_now = event_frame_now
        self.ellipse_last = ellipse_last

        event_points = self.get_all_points(self.event_frame_now)
        boundary_points = self.get_boundary_points(ellipse_last)
        center = (int(ellipse_last[0][0]), int(ellipse_last[0][1]))
        avg_distance = self.cal_average_distance(event_points, center)
        candidate_points = self.get_candidate_points(event_points, center, avg_distance)
        nearest_points = self.get_nearest_points(candidate_points, boundary_points)
        T = self.update_T(nearest_points, candidate_points)
        new_ellipse = self.get_new_ellipse(center, T)

        return new_ellipse

    def cal_distance(self, point_1, point_2):
        """
        Calculate the distance between two points.
        Args:
            point_1: tuple, the first point.
            point_2: tuple, the second point.
        Returns:
            distance: float, the distance between two points.
        """
        distance = np.linalg.norm(np.array(point_1) - np.array(point_2))
        return distance

    def cal_average_distance(self, points, center):
        """
        Calculate the average distance between points and the center.

        Args:
            points: list, the list of points.
            center: tuple, the center point.
        Returns:
            avg_distance: float, the average distance between points and the center.
        """
        distances = [self.cal_distance(p, center) for p in points]
        avg_distance = np.mean(distances)
        return avg_distance

    def get_all_points(self, image):
        """
        Get the coordinates of all the points in the image.

        Args:
            image: numpy.ndarray, the image.
        Returns:
            points_list: list, the list of the coordinates of all the points.
        """
        points_array = np.argwhere(image)
        points_list = [[p[1], p[0]] for p in points_array]
        return points_list

    def get_boundary_points(self, ellipse):
        """
        Get the coordinates of the boundary points of the ellipse.

        Args:
            ellipse: tuple, (x, y, a, b, ang), the parameters of the ellipse.
            x, y: int, the coordinates of the center of the ellipse.
            a, b: int, the major and minor axes of the ellipse, not half of the axes.
            ang: int, the angle of the ellipse.
        Returns:
            boundary_points_list: list, the list of the coordinates of the boundary points.
        """
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = int(ellipse[2])
        startAngle = 0
        endAngle = 360
        delta = 1  # 角度步长，越小越精细

        boundary_points = cv2.ellipse2Poly(
            center, axes, angle, startAngle, endAngle, delta
        )
        boundary_points_list = boundary_points.tolist()

        return boundary_points_list

    def get_candidate_points(
        self, points, center, avg_distance, N=20, lamda_1=0.8, lambda_2=1.2
    ):
        """
        Get the candidate points.

        Args:
            points: list, the list of the coordinates of all the points.
            center: tuple, the center point.
            avg_distance: float, the average distance between points and the center.
            N: int, the number of nearest points to keep.
            lamda_1: float, the lower bound of the distance.
            lambda_2: float, the upper bound of the distance.
        Returns:
            candidate_points: list, the list of the candidate points.
        """
        candidate_points = [
            p
            for p in points
            if lamda_1 * avg_distance
            < self.cal_distance(p, center)
            < lambda_2 * avg_distance
        ]
        # 按与中心的距离排序并保留最近的N个点
        candidate_points = sorted(
            candidate_points, key=lambda p: self.cal_distance(p, center)
        )[:N]
        return candidate_points

    def get_nearest_points(self, points, boundary_points):
        """
        Get the nearest points of the points on the boundary.

        Args:
            points: list, the list of the points.
            boundary_points: list, the list of the boundary points.
        Returns:
            nearest_points: list, the list of the nearest points.
        """
        nearest_points = []
        for p in points:
            distances = [self.cal_distance(p, bp) for bp in boundary_points]
            min_index = np.argmin(distances)
            nearest_points.append(boundary_points[min_index])
        return nearest_points

    def cal_average_displacement(self, nearest_points, candidate_points):
        """
        Calculate the average displacement of the candidate points.

        Args:
            nearest_points: list, the list of the nearest points.
            candidate_points: list, the list of the candidate points.
        Returns:
            avg_delta_Tx: float, the average displacement in x direction.
            avg_delta_Ty: float, the average displacement in y direction.
        """
        delta_Tx = [q[0] - p[0] for p, q in zip(candidate_points, nearest_points)]
        delta_Ty = [q[1] - p[1] for p, q in zip(candidate_points, nearest_points)]
        avg_delta_Tx = np.mean(delta_Tx)
        avg_delta_Ty = np.mean(delta_Ty)
        return avg_delta_Tx, avg_delta_Ty

    def update_candidate_points(self, candidate_points, avg_delta_Tx, avg_delta_Ty):
        """
        Update the candidate points.

        Args:
            candidate_points: list, the list of the candidate points.
            avg_delta_Tx: float, the average displacement in x direction.
            avg_delta_Ty: float, the average displacement in y direction.
        Returns:
            updated_candidate_points: list, the list of the updated candidate points.
        """
        updated_candidate_points = [
            (p[0] + avg_delta_Tx, p[1] + avg_delta_Ty) for p in candidate_points
        ]
        return updated_candidate_points

    def update_T(self, nearest_points, candidate_points, max_iterations=100):
        """
        Update the T.

        Args:
            nearest_points: list, the list of the nearest points.
            candidate_points: list, the list of the candidate points.
            max_iterations: int, the maximum number of iterations to prevent infinite loop.
        Returns:
            T: numpy.ndarray, the updated T.
        """
        T = np.array([0, 0], dtype=np.float32)
        iteration = 0

        while iteration < max_iterations:
            avg_delta_Tx, avg_delta_Ty = self.cal_average_displacement(
                nearest_points, candidate_points
            )
            updated_candidate_points = self.update_candidate_points(
                candidate_points, avg_delta_Tx, avg_delta_Ty
            )
            T += np.array([avg_delta_Tx, avg_delta_Ty])
            if np.linalg.norm([avg_delta_Tx, avg_delta_Ty]) / np.linalg.norm(T) < 0.01:
                break
            candidate_points = updated_candidate_points
            iteration += 1

        return T

    def get_new_ellipse(self, center, T):
        """
        Get the new ellipse.

        Args:
            center: tuple, the center of the ellipse.
            T: numpy.ndarray, the updated T.
        Returns:
            new_ellipse: tuple, the new ellipse.
        """
        new_center = (center[0] - T[0], center[1] - T[1])
        new_ellipse = (new_center, self.ellipse_last[1], self.ellipse_last[2])
        return new_ellipse
