import carla


def filter_vehicle_blueprints(available_models):
    four_wheel_vehicles = [
        bp for bp in available_models
        if bp.id.lower().startswith('vehicle.')
        and 'motorcycle' not in bp.id.lower()
        and 'bike' not in bp.id.lower()
    ]
    return four_wheel_vehicles

# 获取车辆大小的辅助函数
def get_vehicle_size(bp):
    # 通过边界框计算车辆体积
    try:
        bbox = bp.get_attribute('bounding_box')
        if bbox:
            extent = bbox.as_vector()
            return extent.x * extent.y * extent.z
    except:
        pass
    return 0


def is_spawn_point_on_driveway(world, spawn_point):
    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(spawn_point.location,
                                      project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    return waypoint.lane_type == carla.LaneType.Driving


def check_spawn_points(world, spawn_points):
    valid_spawn_points = []
    for i, spawn_point in enumerate(spawn_points):
        is_driveway = is_spawn_point_on_driveway(world, spawn_point)
        if is_driveway:
            valid_spawn_points.append(spawn_point)

    print(f"Found {len(valid_spawn_points)} valid spawn points on driveways out of {len(spawn_points)} total")

    if not valid_spawn_points:
        print("Warning: No valid spawn points found on driveways, using first available spawn point")
        valid_spawn_points.append(spawn_points[0])

    return valid_spawn_points


def set_traffic_lights(world):
    """
    设置并自定义交通信号灯周期
    """
    # 获取所有交通信号灯
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        # 尝试设置信号灯状态
        try:
            traffic_light.freeze(True)  # 冻结信号灯状态
            traffic_light.set_state(carla.TrafficLightState.Green)  # 初始设置为绿灯
        except Exception as e:
            print(f"Error setting traffic light state: {e}")


def reset_traffic_lights(world):
    """
    重置交通信号灯
    """
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    for traffic_light in traffic_lights:
        try:
            traffic_light.freeze(False)  # 解冻信号灯
        except Exception as e:
            print(f"Error resetting traffic light: {e}")