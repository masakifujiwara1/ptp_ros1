import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def CreateMarker(data, id_):
    line_strip = Marker()
    line_strip.header.frame_id = "map"
    line_strip.type = Marker.LINE_STRIP
    line_strip.action = Marker.ADD
    line_strip.scale.x = 0.05
    line_strip.color.a = 1.0
    line_strip.color.b = 1.0
    line_strip.id = id_
    line_strip.pose.orientation.x = 0
    line_strip.pose.orientation.y = 0
    line_strip.pose.orientation.z = 0
    line_strip.pose.orientation.w = 1

    points_marker = Marker()
    points_marker.header.frame_id = "map"
    points_marker.type = Marker.POINTS
    points_marker.action = Marker.ADD
    points_marker.scale.x = 0.1
    points_marker.scale.y = 0.1
    points_marker.color.a = 1.0
    points_marker.color.g = 1.0
    points_marker.id = id_ + 1
    points_marker.pose.orientation.x = 0
    points_marker.pose.orientation.y = 0
    points_marker.pose.orientation.z = 0
    points_marker.pose.orientation.w = 1

    for i in range(8):
        x = float(data[i, 0])
        y = float(data[i, 1])
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.
        line_strip.points.append(point)

    for i in range(8, 20):
        x = float(data[i, 0])
        y = float(data[i, 1])
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.
        points_marker.points.append(point)
    
    return line_strip, points_marker