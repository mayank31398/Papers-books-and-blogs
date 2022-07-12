import argparse
import json

from values import SHIELD_VARIABLES, URL_PROVIDED


def ParseDataType(line: str) -> str:
    for i in SHIELD_VARIABLES:
        if ("![image][{}]".format(i) in line):
            return i
    raise NotImplementedError("Unknown type")


def ParseName(line: str, data_type: str) -> str:
    if (URL_PROVIDED[data_type]):
        return line.split("[{}] [".format(data_type))[1].split("](")[0].strip()
    else:
        # This might have errors
        return line.split("[{}] ".format(data_type))[1].split(" - ")[0].strip()


def ParseURL(line: str, data_type: str, authors: str) -> str:
    if (URL_PROVIDED[data_type]):
        if (authors):
            return line.split("](")[1].split(") - ")[0].strip()
        else:
            return line.split("](")[1].split(")")[0].strip()
    else:
        return None


def ParseAuthors(line: str, data_type: str) -> str:
    if (URL_PROVIDED[data_type]):
        x = line.split(") - ")
    else:
        x = line.split(" - ")

    if (len(x) > 1):
        return x[1].strip()
    else:
        return ""


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=str,
                        required=True, help="Readme file")
    parser.add_argument("--json_dump", type=str,
                        required=True, help="JSON dump file")
    return parser.parse_args()


def main():
    args = ParseArgs()

    json_file = []
    with open(args.readme, "r") as f:
        f = f.readlines()
        for line in f:
            if (line[:2] == "# "):
                topic = line[2:].strip()

            if (not line.startswith("1. !")):
                continue

            datatype = ParseDataType(line)
            name = ParseName(line, datatype)
            authors = ParseAuthors(line, datatype)
            url = ParseURL(line, datatype, authors)
            print(datatype)
            print(name)
            print(url)
            print(authors)

            json_file.append({
                "name": name,
                "url": url,
                "topic": topic,
                "datatype": datatype,
                "authors": authors
            })
    json.dump(json_file, open(args.json_dump, "w"), indent=4)


if (__name__ == "__main__"):
    main()
