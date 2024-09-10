import csv
import json


def read_json_data(file_path):
    """ read json data, [{}, {}, ...]"""
    print("Load json data from path: {}".format(file_path))
    with open(file_path, 'r', encoding='utf-8') as fr:
        dataset = json.load(fr)

    return dataset


def read_jsonl_data(file_path):
    """read jsonl data"""
    print("Load jsonl data from path: {}".format(file_path))
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            data = json.loads(line)
            dataset.append(data)

    return dataset


def read_csv_data(file_path):
    """read csv data"""
    print("Load csv data from path: {}".format(file_path))
    dataset = []

    with open(file_path, 'r', encoding='utf-8') as fr:
        reader = csv.reader(fr)
        for line in reader:
            dataset.append(line)

    return dataset


def write_json_data(file_path, dataset):
    """write json data, [{}, {}, ...]"""
    print("Write json data to path: {}".format(file_path))
    with open(file_path, 'w', encoding='utf-8') as fw:
        json.dump(dataset, fw, indent=4, ensure_ascii=False)

def print_args(args):
    for key in args.__dict__:
        print(key + ': ' + str(args.__dict__[key]))