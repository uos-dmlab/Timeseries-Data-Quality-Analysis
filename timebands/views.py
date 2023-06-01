import sys

from django.http import HttpResponse, HttpResponseServerError
from background_task import background
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
import requests
import json
import csv
import pandas as pd
from django.core.serializers.json import DjangoJSONEncoder
import subprocess

def index(request):
    return HttpResponse("This is time series data quality improver using timebands.")

def run_task(request):
    if request.method == 'POST':  # post
        json_data = json.loads(request.body)  # request.raw_post_data w/ Django < 1.4
        try:
            data = json_data
            data_id = data['id']
            file_url = data['fileUrl']

            file_id = file_url.split('/')[-2]
            dwn_url = 'https://drive.google.com/uc?id=' + file_id  # json data of stock sample

            # receive data from url
            sample = receive_data(dwn_url)

            # save data to timeband data folder
            data_file = "stock.sample_v2"
            sample.to_csv('/django_project/TIMEBAND/data/' + data_file + '.csv', index=False)

            # run timeband
            target_output = subprocess.Popen([sys.executable, "/django_project/TIMEBAND/launcher.py"])
            # vis = pd.read_csv("C:/Users/hs/Desktop/djangoProject2/TIMEBAND/outputs/stock.sample/all/visualize.csv")

            # export output result
            # export_url = '127.0.0.1:8000/api/timebands/run/'
            # post_data = target_output
            # api_call = requests.post(export_url, headers={}, data=data)

        except KeyError:
            HttpResponseServerError("Malformed data!")
        return HttpResponse("Got json data: id: " + str(data_id) + ", file_url: " + file_url)
    else:
        return HttpResponse("not allowed")

# 1. receive data and convert dataframe
def receive_data(file_url):
    # print("Working now")
    df = requests.get(file_url).json()
    js = json.loads(df)
    data = pd.DataFrame(js)
    data = data.iloc[:3000, :]
    # time band code
    return data

# 2. run and save result
def run_timeband(file_url):
    print("Working now")
    df = requests.get(file_url).json()
    data = df[:1000]
    # time band code
    return data

# 경로 확인 필요
# 3. convert csv file to json file
def convert_csv_json(data_file):
    input_file_name = "./TIMEBAND/outputs/" + data_file + "/all/visualize.csv"
    output_file_name = "./TIMEBAND/outputs/" + data_file + "/all/visualize.txt"
    with open(input_file_name, "r", encoding="utf-8", newline="") as input_file, \
            open(output_file_name, "w", encoding="utf-8", newline="") as output_file:
        reader = csv.reader(input_file)
        col_names = next(reader)

        for cols in reader:
            doc = {col_name: col for col_name, col in zip(col_names, cols)}
            print(json.dumps(doc, ensure_ascii=False), file=output_file)

    # Function
    # 1. save data to data directory
    # 2. run and save result
    # 3. convert csv file to json file
    # 4. call qufa api with result json file

# window error : https://stackoverflow.com/questions/63987965/typeerror-argument-of-type-windowspath-is-not-iterable-in-django-python

# migragion : https://stackoverflow.com/questions/65716772/operational-error-no-such-table-background-task-clean-way-to-deploy
## Operational Error: no such table: background_task

# background task : https://stackoverflow.com/questions/65410063/django-background-tasks-wait-for-result

# subprocess method
### https://fedingo.com/how-to-run-python-script-in-django-project/
### https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command

# 외부접속 : https://soyoung-new-challenge.tistory.com/62