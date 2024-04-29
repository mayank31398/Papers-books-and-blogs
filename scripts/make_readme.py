import argparse
import json
import re
from typing import List

from values import METADATA, SHIELD_VARIABLES


def NormalizeTopic(topic: str) -> str:
    return topic.lower().replace(", ", "-").replace(" ", "-")


def GetAllValuesByField(json_dump: List[dict], field: str) -> List[str]:
    x = set()
    for i in json_dump:
        if i[field]:
            if type(i[field]) == list:
                i[field].sort()
                for j in i[field]:
                    x.add(j)
            else:
                x.add(i[field])

    x = list(x)
    x.sort()

    return x


def PreprocessNames(json_file: List[dict]) -> List[dict]:
    pattern = re.compile("\^{\w*}")
    for i in json_file:
        x = pattern.findall(i["name"])
        for j in x:
            exponent = j[2 : len(j) - 1]
            i["name"] = i["name"].replace(j, "<sup>{}</sup>".format(exponent))

    return json_file


def GroupByTopics(json_file: List[dict], topics: List[str]) -> List[dict]:
    grouped_json = {}
    for topic in topics:
        grouped_json[topic] = []
        for i in json_file:
            if i["topic"] == topic:
                grouped_json[topic].append(i)

    for topic in grouped_json:
        grouped_json[topic].sort(key=lambda x: x["name"].lower())

    return grouped_json


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=str, required=True, help="Readme file")
    parser.add_argument("--json_dump", type=str, required=True, help="JSON dump file")
    parser.add_argument("--group_by_datatype", action="store_true", help="Group by data type")
    parser.add_argument("--add_shields", action="store_true", help="Add shields")
    return parser.parse_args()


def MakeElement(json_obj: json, add_shield: bool) -> str:
    element = "1. "

    if json_obj["url"]:
        element += "[{}]({})".format(json_obj["name"], json_obj["url"])
    else:
        element += "{}".format(json_obj["name"])

    # if json_obj["authors"]:
    #     element += "  \n*{}*".format(json_obj["authors"])

    if add_shield:
        element += "  \n![image][{}]".format(json_obj["datatype"])
        if json_obj["keywords"]:
            # if json_obj["venue"]:
            #     element += " ![image][{}]".format(json_obj["venue"])
            for k in json_obj["keywords"]:
                element += " ![image][{}]".format(k)

    # if (json_obj["metadata"]):
    #     element += "\n> {}".format(json_obj["metadata"])

    element += "\n"

    return element


def main():
    args = ParseArgs()

    json_file = json.load(open(args.json_dump, "r"))
    json_file = PreprocessNames(json_file)
    topics = GetAllValuesByField(json_file, "topic")
    keywords = GetAllValuesByField(json_file, "keywords")
    grouped_json = GroupByTopics(json_file, topics)

    with open(args.readme, "w") as f:
        f.write(METADATA + "\n\n")
        for shield in SHIELD_VARIABLES:
            f.write(SHIELD_VARIABLES[shield] + "\n")

        for keyword in keywords:
            f.write(
                f"[{keyword}]: https://img.shields.io/static/v1?label=&message={keyword.replace(' ', '%20')}&color=blue\n"
            )
        f.write("\n")

        # for venue in venues:
        #     f.write(
        #         f"[{venue}]: https://img.shields.io/static/v1?label=&message={venue.replace(' ', '%20')}&color=grey\n"
        #     )
        # f.write("\n")

        f.write("### Table of contents\n")
        for topic in topics:
            f.write("- [{}](#{})\n".format(topic, NormalizeTopic(topic)))
        f.write("\n")

        for topic in topics:
            f.write("# {}\n".format(topic))

            if args.group_by_datatype:
                for datatype in SHIELD_VARIABLES:
                    for i in grouped_json[topic]:
                        if i["datatype"] == datatype:
                            f.write(MakeElement(i, args.add_shields))
            else:
                for i in grouped_json[topic]:
                    f.write(MakeElement(i, args.add_shields))
            f.write("\n")


if __name__ == "__main__":
    main()
