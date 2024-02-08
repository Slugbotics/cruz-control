# This file needs work
import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

blueprint_library = world.get_blueprint_library()


bp = random.choice(blueprint_library.filter('vehicle'))

# A blueprint contains the list of attributes that define a vehicle's
# instance, we can read them and modify some of them. For instance,
# let's randomize its color.
if bp.has_attribute('color'):
    color = random.choice(bp.get_attribute('color').recommended_values)
    bp.set_attribute('color', color)

# Now we need to give an initial transform to the vehicle. We choose a
# random transform from the list of recommended spawn points of the map.
transform = random.choice(world.get_map().get_spawn_points())

# So let's tell the world to spawn the vehicle.
ego_vehicle = world.spawn_actor(bp, transform)

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
