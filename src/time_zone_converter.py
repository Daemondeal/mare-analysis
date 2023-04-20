from utils import read_csv, write_dataframe
from collections import deque
import datetime


def fix_tz(data, timezone):
    fixed = {}
    for k, v in data.items():
        if k == "time":
            arr = [x - datetime.timedelta(hours=-timezone) for x in v]
        elif k == "index":
            arr = [(x - timezone) % len(v) for x in v]
        else:
            arr = v

        col = deque(arr)
        col.rotate(-timezone)
        fixed[k] = list(col)

    return fixed

def fix_csv(csv_file, timezone):
    fixed = fix_tz(read_csv(csv_file), timezone)
    csv_without_extension = csv_file[:-4]
    write_dataframe(csv_without_extension + "_fixed.csv", fixed)

def main():
    fix_csv("../data/lipari/wind_and_waves_2.csv", 1)


main()