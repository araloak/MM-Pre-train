import base64, json, requests,time,os
import argparse

image_path = ""
json_path = ""
parser = argparse.ArgumentParser(description='extract CNN pooling features from images')

parser.add_argument('--start-num', required=True,type = int,
                    help='origin data directory.')
parser.add_argument('--end-num', required=True,type = int,
                    help='origin data directory.')
parser.add_argument('--api-index', required=True,type = int,
                    help='origin data directory.')
args = parser.parse_args()
def iterate_img_dir(img_dir):
    """iterate all images inside img_dir"""
    for i in range(1662888):
        output.append(image_path+str(i)+".jpg")
    return output

def img2base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
        image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64

def use_api(key,secret,start_num,end_num):
    FACE_PLUS_PLUS_API_KEY = key
    FACE_PLUS_PLUS_API_SECRET = secret
    FACESET_TOKEN = ""

    DETECT_API_URL = ""
    SEARCH_API_URL = ""
    ANALYZE_API_URL = ""
    ADD_FACESET_API_URL = ""

    def fac_detect(image_base64,index):
        data = {
        "api_key": FACE_PLUS_PLUS_API_KEY,
        "api_secret": FACE_PLUS_PLUS_API_SECRET,
        "image_base64": image_base64,
        "return_attributes": "emotion"

        }
        r = requests.post(DETECT_API_URL,data = data)
        return_data = r.json()
        #print(type(return_data))
        #print(return_data)
        file = open(json_path+str(index)+".json","w",encoding="utf8")
        json.dump(return_data, file,ensure_ascii=False)

    for index in range(args.start_num,args.end_num):
        try:
            fac_detect(img2base64(image_path+str(index)+".jpg"),index)
        except Exception as e:
            print(index,"failed")
            print(e)
        time.sleep(1)

        if index%10000==0:
            print(index)
        #print(index)
    print("done")
if __name__ == '__main__':
    FACE_PLUS_PLUS_API_KEY1 = ""
    FACE_PLUS_PLUS_API_SECRET1 = ""

    FACE_PLUS_PLUS_API_KEY2 = ""
    FACE_PLUS_PLUS_API_SECRET2 = ""

    FACE_PLUS_PLUS_API_KEY3 = ""
    FACE_PLUS_PLUS_API_SECRET3 = ""

    FACE_PLUS_PLUS_API_KEY4 = ""
    FACE_PLUS_PLUS_API_SECRET4 = ""

    FACE_PLUS_PLUS_API_KEY5 = ""
    FACE_PLUS_PLUS_API_SECRET5 = ""

    FACE_PLUS_PLUS_API_KEY6 = ""
    FACE_PLUS_PLUS_API_SECRET6 = ""

    FACE_PLUS_PLUS_API_KEY7 = ""
    FACE_PLUS_PLUS_API_SECRET7 = ""

    FACE_PLUS_PLUS_API_KEY8 = ""
    FACE_PLUS_PLUS_API_SECRET8 = ""

    FACE_PLUS_PLUS_API_KEY9 = ""
    FACE_PLUS_PLUS_API_SECRET9 = ""

    FACE_PLUS_PLUS_API_KEY10 = ""
    FACE_PLUS_PLUS_API_SECRET10 = ""

    FACE_PLUS_PLUS_API_KEY11 = ""
    FACE_PLUS_PLUS_API_SECRET11 = ""
    
    FACE_PLUS_PLUS_API_KEY12 = ""
    FACE_PLUS_PLUS_API_SECRET12 = ""
    
    FACE_PLUS_PLUS_API_KEY13 = ""
    FACE_PLUS_PLUS_API_SECRET13 = ""

    FACE_PLUS_PLUS_API_KEY14 = ""
    FACE_PLUS_PLUS_API_SECRET14 = ""

    FACE_PLUS_PLUS_API_KEY15 = ""
    FACE_PLUS_PLUS_API_SECRET15 = ""

    
    api_keys = [FACE_PLUS_PLUS_API_KEY1,FACE_PLUS_PLUS_API_KEY2,FACE_PLUS_PLUS_API_KEY3,FACE_PLUS_PLUS_API_KEY4,FACE_PLUS_PLUS_API_KEY5,FACE_PLUS_PLUS_API_KEY6,FACE_PLUS_PLUS_API_KEY7,FACE_PLUS_PLUS_API_KEY8,FACE_PLUS_PLUS_API_KEY9,FACE_PLUS_PLUS_API_KEY10,FACE_PLUS_PLUS_API_KEY11,FACE_PLUS_PLUS_API_KEY12,FACE_PLUS_PLUS_API_KEY13,FACE_PLUS_PLUS_API_KEY14,FACE_PLUS_PLUS_API_KEY15]
    api_secrets = [FACE_PLUS_PLUS_API_SECRET1,FACE_PLUS_PLUS_API_SECRET2,FACE_PLUS_PLUS_API_SECRET3,FACE_PLUS_PLUS_API_SECRET4,FACE_PLUS_PLUS_API_SECRET5,FACE_PLUS_PLUS_API_SECRET6,FACE_PLUS_PLUS_API_SECRET7,FACE_PLUS_PLUS_API_SECRET8,FACE_PLUS_PLUS_API_SECRET9,FACE_PLUS_PLUS_API_SECRET10,FACE_PLUS_PLUS_API_SECRET11,FACE_PLUS_PLUS_API_SECRET12,FACE_PLUS_PLUS_API_SECRET13,FACE_PLUS_PLUS_API_SECRET14,FACE_PLUS_PLUS_API_SECRET15]

    use_api(api_keys[args.api_index],api_secrets[args.api_index],args.start_num,args.end_num) 

