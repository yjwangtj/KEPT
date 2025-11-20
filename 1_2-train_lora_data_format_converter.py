import json, argparse

system = '''You are an autonomous driving trajectory prediction assistant.
You will be given:
1. Seven sequential front-camera images taken at 3.0s, 2.5s, 2.0s, 1.5s, 1.0s, 0.5s, and 0.0s (current time) before now.
2. The past vehicle states including both position (x, y) and speed (km/h) in the current coordinate system (current position is (0, 0)).
3. The current vehicle speed (in km/h).
4. An optional navigation waypoint (soft target), which represents a future desired direction but does not require reaching it within 3 seconds.

Here are some considerations for trajectory planning: 
1. It must be noted that there must be no collision.
2. The velocity of the trajectory should be smooth, and do not change too much if it is not necessary
3. The trajectory needs to be generated on the road where the vehicle can travel, not in inaccessible areas such as crosswalks and barricades.
4. The output coordinates and vehicle velocity are reserved to one decimal place.
5. The coordinates that the vehicle needs to pass through do not need to arrive in 3 seconds, it is only a guide in a direction.
Please output the destination point of the vehicle coordinate system for the next 1,2,3 seconds and the velocity of the vehicle when it reaches that point, respectively. The definition of vehicle coordinate system is the same as the definition of vehicle coordinate system at current time. 
Output strictly in JSON format like: 
##################
{"0.5 seconds": {"x": 1.1, "y": 0.2, "velocity": 23.5}, "1.0 seconds": {"x": 2.1, "y": 0.5, "velocity": 26.5}, "1.5 seconds": {"x": 3.9, "y": 0.8, "velocity": 30.8}, "2.0 seconds": {"x": 5.8, "y": 0.8, "velocity": 30.3}, "2.5 seconds": {"x": 7.8, "y": 0.7, "velocity": 30.2}, "3.0 seconds": {"x": 9.5, "y": 0.6, "velocity": 28.1}}
##################
, with each value rounded to 1 decimal place. '''

parser = argparse.ArgumentParser()
parser.add_argument("--input",  dest="in_path",  required=True)
parser.add_argument("--output", dest="out_path", required=True)
parser.add_argument("--status", action="store_true", help="set it True to add ego status")
args = parser.parse_args()

with open(args.in_path, "r", encoding="utf-8") as f:
    datas = json.load(f)

data_format = []
for data in datas:
    hist = json.loads(data["history_trajectories"])
    question = (
        "Image 1: <image>\nImage 2: <image>\nImage 3: <image>\nImage 4: <image>\n"
        "Image 5: <image>\nImage 6: <image>\nImage 7: <image>\n"
        "Here are seven sequential front-camera images taken at corresponding moments before now.\n"
        "Past vehicle's states:\n"
        f"3.0s ago: {{ \"x\": {hist['3.0 seconds ago']['x']}, \"y\":{hist['3.0 seconds ago']['y']},\"velocity\": {hist['3.0 seconds ago']['velocity']} }}\n"
        f"2.0s ago: {{ \"x\": {hist['2.0 seconds ago']['x']}, \"y\":{hist['2.0 seconds ago']['y']},\"velocity\": {hist['2.0 seconds ago']['velocity']} }}\n"
        f"1.0s ago: {{ \"x\": {hist['1.0 seconds ago']['x']}, \"y\": {hist['1.0 seconds ago']['y']},\"velocity\": {hist['1.0 seconds ago']['velocity']} }}\n"
        f"Navigation waypoint in current coordinate system: (\"x\": {data['navi']['x']} , \"y\": {data['navi']['y']})\n"
    )
    if args.status:
        question += (
            f"The current velocity of ego is {data['status']['speed_mps']} km/h.\n"
            f"The current forward acceleration of the vehicle is {data['status']['accel_mps2_xyz'][0]}m/s**2; "
            f"The current acceleration of the vehicle to the left is {data['status']['accel_mps2_xyz'][1]}m/s**2.\n"
            f"The current yaw angle of the vehicle is {data['status']['yaw_rad']} rad\n"
        )
    question += "\nPlease predict the next 3 seconds trajectory."

    data_format.append({
        "system": system,
        "images": data["images"],
        "messages": [
            {"content": question, "role": "user"},
            {"content": data["new_trajectories"], "role": "assistant"}
        ]
    })

with open(args.out_path, "w", encoding="utf-8") as res:
    json.dump(data_format, res, indent=4, ensure_ascii=False)

