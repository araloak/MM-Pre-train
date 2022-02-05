a = "nohup python API.py --start-num 0 --end-num 10  > ./API.log 2>&1 &"
cmd = []
real_api_index = 0
for api_index in range(15,30):
    start_index = 60000 + 640*api_index
    end_index = start_index+640
    cmd.append("nohup python API2.py --start-num {start} --end-num {end} --api-index {api} > ./API2_{api}.log 2>&1 &".format(start=start_index,end=end_index,api=real_api_index) )
    real_api_index+=1
#print(cmd)

with open("./API.sh","w",encoding = "utf8") as f:
    for each in cmd:
        f.write(each+"\n")

