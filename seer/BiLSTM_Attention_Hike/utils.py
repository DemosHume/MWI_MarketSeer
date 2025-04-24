import sqlite3
import pandas as pd
from datetime import datetime
import pytz


def query(q_string, fname='market.db'):
    conn = sqlite3.connect(fname)
    data = pd.read_sql_query(q_string, conn)
    conn.close()
    return data

def timestamp_to_beijing_time(ts):
    beijing_tz = pytz.timezone('Asia/Shanghai')
    dt = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(beijing_tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S')
