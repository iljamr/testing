import math

import rospy
import sensor_msgs.msg
from sensor_msgs import point_cloud2


class BagRelay:
    def __init__(self):
        rospy.init_node("relay")

        # Constants
        self.THETA = math.radians(35)
        self.FIELD_NAMES = ("x", "y", "z", "intensity", "label")

        # Read parameters
        self.input_topic = rospy.get_param("~input_topic", "/os_cloud_node/points")
        self.output_topic = rospy.get_param(
            "~output_topic", "/os_cloud_node/points_relay"
        )

        # Subscribers
        rospy.Subscriber(
            self.input_topic, sensor_msgs.msg.PointCloud2, self.relay_callback
        )

        # Publishers
        self.relay_pub = rospy.Publisher(
            self.output_topic, sensor_msgs.msg.PointCloud2, queue_size=10
        )

    def transformed_points(self, data):
        transformed_points = []
        for point in point_cloud2.read_points(data, field_names=self.FIELD_NAMES):
            x_orig, y_orig, z_orig, intensity, *_ = point

            x = math.cos(self.THETA) * x_orig - math.sin(self.THETA) * y_orig + 0.12
            y = -math.sin(self.THETA) * x_orig - math.cos(self.THETA) * y_orig + 0.0
            z = -z_orig - 0.14

            # TODO make useful label
            label = 0 if z > 0 else 1

            transformed_points.append((x, y, z, intensity, label))

        return transformed_points

    def extended_with_lables(self, data):
        new_fields = [
            point_cloud2.PointField("label", 36, point_cloud2.PointField.UINT8, 1)
        ]
        selected_fields = [
            field for field in data.fields if field.name in self.FIELD_NAMES
        ]
        return selected_fields + new_fields

    def relay_callback(self, data):
        rospy.loginfo(f"Relaying data from {self.input_topic} to {self.output_topic}")

        transformed_points = self.transformed_points(data)
        extended_filelds = self.extended_with_lables(data)

        transformed_data = point_cloud2.create_cloud(
            data.header, extended_filelds, transformed_points
        )

        self.relay_pub.publish(transformed_data)


if __name__ == "__main__":
    try:
        relay = BagRelay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
