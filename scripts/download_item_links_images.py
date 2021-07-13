import json
import requests
import shutil
import os

output_dir = "../images/item_links/training/6L/"

file = open('./data.json')
data = json.load(file)


def get_next_num():
    i = 1
    while os.path.exists("{}.png".format(i)):
        i = i + 1

    return i


def run():
    num = get_next_num()

    for item in data["result"]:
        image_url = item["item"]["icon"]
        print("URL: ", image_url)

        filename = "{}{}.png".format(output_dir, num)
        num = num + 1

        r = requests.get(image_url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)


run()
