import requests
import socket
import argparse
import os
parser = argparse.ArgumentParser(description='extract CNN pooling features from images')

parser.add_argument('--start-index', required=True,type = int,
                    help='origin data directory.')
parser.add_argument('--num', required=False,default=10000,type = int,
                    help='origin data directory.')
parser.add_argument('--output-dir', required=False,default='./img/',
                    help='output directory.')
parser.add_argument('--tread-index', required=True,type =int,
                    help='output directory.')
args = parser.parse_args()

timeout = 20
socket.setdefaulttimeout(timeout)
 
 
urls = []
with open('./urls.txt') as f:
    for i in f.readlines():
        if i != '':
            urls.append(i)
        else:
            pass
 
 
user_agent = ''
headers = {
    'User-Agent': user_agent
}
bad_url = []

count = 0
for index,url in enumerate(urls):
    if index >= args.start_index and index<args.start_index+args.num:
        url.rstrip('\n')
    #print(url)
        try:
            pic = requests.get(url, headers=headers)
            with open('./img/%d.jpg' % index, 'wb') as f:
                f.write(pic.content)
                f.flush()
            if count%1000==0:
                print('pic %d' % count)
            count += 1
        except Exception as e:
            print(Exception, ':', e)
            bad_url.append([index,url])
print('got all photos that can be got')
 
 
with open('./'+str(args.tread_index)+'_bad_url.txt', 'w') as f:
    for [index,url] in bad_url:
        f.write(str(index)+"\t"+url)
        f.write('\n')
    print('saved bad urls')
