import json, os
BASE = json.load(open(os.path.expanduser("~/projects/NetGPT_work/models/gpt2/config.json")))
OUT = os.path.expanduser("~/projects/NetGPT_work/configs"); os.makedirs(OUT, exist_ok=True)
def emit(name, layers, dff, heads):
    c = dict(BASE); c["layers_num"]=layers; c["feedforward_size"]=dff; c["heads_num"]=heads
    json.dump(c, open(f"{OUT}/{name}.json","w"), indent=2)
emit("teacher", 12, 3072, 12); emit("slim", 1, 256, 12); emit("scramble", 6, 3072, 8)
for L in [1,2,3,4,5,6]: emit(f"depth_{L}", L, 3072, 12)
for W in [512,768,1024,1280,1536,2048,2560,3072]: emit(f"width_{W}", 6, W, 12)
for W in [1,2,4,8,16,32,64,128,256]: emit(f"w1L_{W}", 1, W, 12)
print("OK, configs regeneres depuis models/gpt2/config.json")
