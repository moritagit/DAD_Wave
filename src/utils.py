# -*- coding: utf-8 -*-


from pathlib import Path

import codecs
import csv
import pandas as pd

import json
import yaml


def load_table(path, encoding='utf-8-sig'):
    """Load table file and make Dataframe from it.

    Args:
        path (str ot pathlib.Path): Path to config file.
            Extentions are supposed to be:
                * csv
                * xlsx
                * xls
        encoding (str): Encoding of the file.
            Defaults to 'utf-8-sig'.

    Returns:
        df (pandas.DataFrame): Dataframe made from the file.
    """
    path = Path(path)
    extention = path.suffix
    if extention == '.csv':
        f = codecs.open(str(path), 'r', encoding=encoding,)
        header = f.readline().strip()
        if header and (header[0] == '\ufeff'):  # eliminate BOM
            header = header[1:]
        header = header.split(',')

        rows = []
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()

        df = pd.DataFrame(rows, columns=header)
        df = df.replace({'': None})

        # Maybe pandas can't handle UTF-8 CSV file with BOM and zenkaku file name.
        # engine=python -> can't read first line (header) rightly.
        #df = pd.read_csv(str(path), encoding=encoding, engine='python',)
    elif (extention == '.xlsx') or (extention == '.xls'):
        df = pd.read_excel(str(path), encoding=encoding)
    else:
        raise ValueError("File extention is invalid! Use 'csv', 'xlsx', or 'xls' as a config file.")
    return df


def config_reader(filepath):
    with open(str(filepath), 'r', encoding='utf-8') as f:
        parser = edict(yaml.load(f))
    return parser


class ConfigJSON(dict):
    """Configuration for model.
    Args:
        path (str ot pathlib.Path): Path to configuration file (json).
    Attributes:
        data: data.
    """
    class _Dict(dict):
        __getattr__ = dict.__getitem__

    def __init__(self, path):
        with open(str(path), 'r') as fin:
            self.data = json.load(fin, object_hook=Config._Dict)

    def to_dict(self):
        """Convert data to dict.
        Returns:
            data_dict (dict)
        """
        data_str = str(self.data)
        data_str = data_str.replace("'", '"')
        data_str = data_str.replace('True', 'true')
        data_str = data_str.replace('False', 'false')
        data_dict = json.loads(data_str)
        return data_dict
