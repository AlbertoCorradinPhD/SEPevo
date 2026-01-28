import json

def clean_json_file(file_path, output_path):
	
	text = open(file_path, "r", encoding="utf-8").read()
	decoder = json.JSONDecoder()

	objects = []
	idx = 0
	n = len(text)

	while idx < n:
		obj, idx = decoder.raw_decode(text, idx)
		objects.append(obj)

		# skip whitespace between objects
		while idx < n and text[idx].isspace():
			idx += 1
	objects=objects[1:]
	print(f"Parsed {len(objects)} objects")

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(objects, f, indent=2, ensure_ascii=False)
	
	return objects
