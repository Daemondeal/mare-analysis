import re
from datetime import datetime

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATE_AND_TIME_ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
DATE_AND_TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
COMPRESSED_DATE_PATTERN = re.compile(r"^\d{8}:\d{4}$")
DATE_WITH_SLASHES_PATTERN = re.compile(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{1,2}$")


def is_date_and_time(value: str) -> bool:
    return DATE_AND_TIME_PATTERN.match(value) is not None


def is_iso_date(value: str) -> bool:
    return ISO_DATE_PATTERN.match(value) is not None


def is_iso_date_and_time(value: str) -> bool:
    return DATE_AND_TIME_ISO_PATTERN.match(value) is not None


def is_compressed_date(value: str) -> bool:
    return COMPRESSED_DATE_PATTERN.match(value) is not None


def is_date_with_slashes(value: str) -> bool:
    return DATE_WITH_SLASHES_PATTERN.match(value) is not None


def is_integer(string: str) -> bool:
    try:
        _ = int(string)
        return True
    except ValueError:
        return False


def is_float(string: str) -> bool:
    try:
        _ = float(string)
        return True
    except ValueError:
        return False


def write_dataframe(filename: str, frame: dict):
    with open(filename, "w") as out_file:
        index = [x for x in frame.keys()]
        cardinality = len(frame[index[0]])

        out_file.write(",".join(index) + "\n")

        for i in range(cardinality):
            row = [str(frame[x][i]) for x in index]
            out_file.write(",".join(row) + "\n")


def read_csv(filename: str, *, ignore=None) -> dict:
    if ignore is None:
        ignore = []

    with open(filename, "r") as file:
        index = file.readline().strip().split(",")
        result = {}

        indices_to_ignore = []
        for i, item in enumerate(index):
            if item in ignore:
                indices_to_ignore.append(i)
            else:
                result[item] = []

        for line in file:
            for i, item in enumerate(line.strip().split(",")):
                if i in indices_to_ignore:
                    continue

                # Checking type
                if is_compressed_date(item):
                    value = datetime.strptime(item, "%Y%m%d:%H%M")
                elif is_integer(item):
                    value = int(item)
                elif is_float(item):
                    value = float(item)
                elif is_date_with_slashes(item):
                    value = datetime.strptime(item, "%m/%d/%Y %H:%M")
                elif is_date_and_time(item):
                    value = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")
                elif is_iso_date_and_time(item):
                    value = datetime.strptime(item, "%Y-%m-%dT%H:%M:%S")
                elif is_iso_date(item):
                    value = datetime.strptime(item, "%Y-%m-%d")
                else:
                    value = item

                result[index[i]].append(value)

    return result


def test():
    frame = read_csv("test.csv")
    for key in frame:
        print(frame[key][8759])

if __name__ == "__main__":
    test()