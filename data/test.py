import json

def write_array_to_file(array, filename):
    with open(filename, 'w') as file:
        for item in array:
            file.write(str(item) + '\n')

with open('sarcasm_data.json', 'r', encoding="utf8") as file:
    data = json.load(file)

write_array_to_file([data[i]['utterance'] for i in data], 'bert-input.txt')