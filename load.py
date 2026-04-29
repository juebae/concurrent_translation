import json

with open("/home/zubair/disso/datasets/flores_test/samples.json") as f:
    data = json.load(f)
print(len(data))
