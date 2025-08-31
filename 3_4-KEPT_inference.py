import torch
import json, argparse
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

'''
Usage:
    python 3_4-KEPT_inference.py \
      --model_dir /path/to/model \
      --retrieval /path/to/3_****_top-k.json \
      --val /path/to/1-****_val.json \
      --db /path/to/1-****_train.json \
      --topk 3 \
      --out /path/to/4-**output**.json

Usage:
    python 3_4-KEPT_inference.py \
      --model_dir /path/to/model \
      --retrieval /path/to/3_****_top-k.json \
      --val /path/to/1-****_val.json \
      --db /path/to/1-****_train.json \
      --topk 3 \
      --out /path/to/4-**output**.json
      --withstatus

'''
def ordinal(n:int)->str:
    return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"

def build_system_prompt(topk:int)->str:
    return (
        "You are an autonomous driving trajectory prediction assistant.\n"
        "You will be given:\n"
        "1. Seven sequential front-camera images taken at 3.0s, 2.5s, 2.0s, 1.5s, 1.0s, 0.5s, and 0.0s (current time) before now.\n"
        "2. The past vehicle states including both position (x, y) and speed (km/h) in the current coordinate system (current position is (0, 0)).\n"
        "3. The current vehicle speed (in km/h).\n"
        "4. An optional navigation waypoint (soft target), which represents a future desired direction but does not require reaching it within 3 seconds.\n"
        f"Before you start, I will provide {topk} similar traffic scene(s) which will assist you to better understand your task and my request. "
        "Please note that I am providing separate but similar scenes for you to understand and learn from. "
        "You need to think about the similarities and differences between the reference scene(s) and the target scene (which is for you to reason).\n"
        "Here are some considerations for trajectory planning: \n"
        "1. It must be noted that there must be no collision.\n"
        "2. The velocity of the trajectory should be smooth, and do not change too much if it is not necessary\n"
        "3. The trajectory needs to be generated on the road where the vehicle can travel, not in inaccessible areas such as crosswalks and barricades.\n"
        "4. The output coordinates and vehicle velocity are reserved to one decimal place.\n"
        "5. The coordinates that the vehicle needs to pass through do not need to arrive in 3 seconds, it is only a guide in a direction.\n"
        "Please output the destination point of the vehicle coordinate system for the next 1,2,3 seconds and the velocity of the vehicle when it reaches that point, respectively. "
        "The definition of vehicle coordinate system is the same as the definition of vehicle coordinate system at current time. \n"
        "Output strictly in JSON format like: \n"
        "##################\n"
        "{\"0.5 seconds\": {\"x\": 1.1, \"y\": 0.2, \"velocity\": 23.5}, \"1.0 seconds\": {\"x\": 2.1, \"y\": 0.5, \"velocity\": 26.5}, "
        "\"1.5 seconds\": {\"x\": 3.9, \"y\": 0.8, \"velocity\": 30.8}, \"2.0 seconds\": {\"x\": 5.8, \"y\": 0.8, \"velocity\": 30.3}, "
        "\"2.5 seconds\": {\"x\": 7.8, \"y\": 0.7, \"velocity\": 30.2}, \"3.0 seconds\": {\"x\": 9.5, \"y\": 0.6, \"velocity\": 28.1}}\n"
        "##################\n, with each value rounded to 1 decimal place."
    )

def make_ref_block(idx:int, db_item:dict)->list:
    RA_images = db_item["images"]
    RA_new_traj = db_item["new_trajectories"]
    RA_history = json.loads(db_item["history_trajectories"])
    RA_velocity = db_item["velocity"]
    RA_navi = db_item["navi"]

    text = (
        f"In the reference scene #{idx+1},"
        "<image> is the scene in front of the vehicle 3.0 seconds ago."
        "<image> is the scene in front of the vehicle 2.5 seconds ago."
        "<image> is the scene in front of the vehicle 2.0 seconds ago."
        "<image> is the scene in front of the vehicle 1.5 seconds ago."
        "<image> is the scene in front of the vehicle 1.0 second ago."
        "<image> is the scene in front of the vehicle 0.5 second ago."
        "<image> is the scene in front of the vehicle at current time(0.0s). "
        f"The velocity of the vehicle is now {RA_velocity} km/h. Located at vehicle coordinate system [0,0]. "
        "In the vehicle coordinate system, the state of the vehicle 3 seconds ago is"
        f"{{ \"x\": {RA_history['3.0 seconds ago']['x']}, \"y\": {RA_history['3.0 seconds ago']['y']}, \"velocity\": {RA_history['3.0 seconds ago']['velocity']} }};"
        " the state of the vehicle 2 seconds ago is "
        f"{{ \"x\": {RA_history['2.0 seconds ago']['x']}, \"y\": {RA_history['2.0 seconds ago']['y']}, \"velocity\": {RA_history['2.0 seconds ago']['velocity']} }};"
        " the state of the vehicle 1 second ago is "
        f"{{ \"x\": {RA_history['1.0 seconds ago']['x']}, \"y\": {RA_history['1.0 seconds ago']['y']}, \"velocity\": {RA_history['1.0 seconds ago']['velocity']} }}\n"
        f"Navigation waypoint in current coordinate system is (\"x\": {RA_navi['x']} , \"y\": {RA_navi['y']})\n\n"
        "The ground truth future trajectory of the ego vehicle in this reference scene is\n"
        f"{RA_new_traj}"
    )

    content = [{"type": "image", "image": RA_images[i]} for i in range(7)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]

def make_target_block(val_item:dict, withstatus:bool, k:int)->list:
    images = val_item["images"]
    hist = json.loads(val_item["history_trajectories"])
    navi = val_item["navi"]
    ord_str = ordinal(k+1)
    text = (
        f"Now here comes the {ord_str} scene for you to predict.\n"
        "Image 1: <image>\nImage 2: <image>\nImage 3: <image>\nImage 4: <image>\nImage 5: <image>\nImage 6: <image>\nImage 7: <image>\n"
        "Here are seven sequential front-camera images taken at corresponding moments before now.\n"
        "Past vehicle's states:\n"
        f"3.0s ago: {{ \"x\": {hist['3.0 seconds ago']['x']}, \"y\": {hist['3.0 seconds ago']['y']}, \"velocity\": {hist['3.0 seconds ago']['velocity']} }}\n"
        f"2.0s ago: {{ \"x\": {hist['2.0 seconds ago']['x']}, \"y\": {hist['2.0 seconds ago']['y']}, \"velocity\": {hist['2.0 seconds ago']['velocity']} }}\n"
        f"1.0s ago: {{ \"x\": {hist['1.0 seconds ago']['x']}, \"y\": {hist['1.0 seconds ago']['y']}, \"velocity\": {hist['1.0 seconds ago']['velocity']} }}\n"
        f"Navigation waypoint in current coordinate system: (\"x\": {navi['x']} , \"y\": {navi['y']})\n"
    )
    if withstatus:
        text += (
            f"The current velocity of ego is {val_item['status']['speed_mps']} km/h.\n"
            f"The current forward acceleration of the vehicle is {val_item['status']['accel_mps2_xyz'][0]}m/s**2; "
            f"The current acceleration of the vehicle to the left is {val_item['status']['accel_mps2_xyz'][1]}m/s**2.\n"
            f"The current yaw angle of the vehicle is {val_item['status']['yaw_rad']} rad\n"
        )
    text += "\n\nPredict the next 3 seconds trajectory."

    content = [{"type": "image", "image": images[i]} for i in range(7)]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]

def main():
    ap = argparse.ArgumentParser(description="KEPT inference")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--withstatus", action="store_true", help="Only when set, append ego status to the target message")
    args = ap.parse_args()
    tries = min(3, max(1, args.retries))

    min_pixels, max_pixels = 256*28*28, 1280*28*28
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype="bfloat16",
        attn_implementation="flash_attention_2", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model_dir, size={"shortest_edge":min_pixels, "longest_edge":max_pixels}
    )


    with open(args.retrieval, "r", encoding="utf-8") as f: retrieval = json.load(f)
    with open(args.val, "r", encoding="utf-8") as f: vals = json.load(f)
    with open(args.db, "r", encoding="utf-8") as f: db = json.load(f)
    system = build_system_prompt(args.topk)

    def chat(i:int):
        ref_indices = retrieval[i]["retrieved_db_indices"][:args.topk]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for r_idx, db_idx in enumerate(ref_indices):
            messages += make_ref_block(r_idx, db[db_idx])
        messages += make_target_block(vals[i], args.withstatus, len(ref_indices))

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        trimmed = [o[len(iid):] for iid, o in zip(inputs.input_ids, out_ids)]
        return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def same_keys(pred_text, gt_text):
        try:
            p = json.loads(pred_text[0]); g = json.loads(gt_text)
            return set(p.keys()) == set(g.keys())
        except: return False

    N = len(vals)
    res = [None]*N
    for i in tqdm(range(N), desc="Infer", unit="item"):
        gt = vals[i]["new_trajectories"]
        ok = False
        for _ in range(tries):
            try:
                pred = chat(i)
                if same_keys(pred, gt):
                    res[i] = {"validation_id": i, "response": pred, "gt": gt}
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            res[i] = {"validation_id": i, "response": None, "gt": gt}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    print(f"total={N}, success={sum(1 for x in res if x['response'] is not None)}")

if __name__ == "__main__":
    main()
