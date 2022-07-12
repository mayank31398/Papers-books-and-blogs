import argparse
import json

from values import METADATA, SHIELD_VARIABLES

def NormalizeTopic(topic: str) -> str:
    return topic.lower().replace(", ", "-").replace(" ", "-")

def GetAllTopics(json_dump: list[dict]) -> list[str]:
    topics = set()
    for i in json_dump:
        topics.add(i["topic"])

    topics = list(topics)
    topics.sort()

    return topics


def GroupByTopics(json_file: list[dict], topics: list[str] = None) -> list[dict]:
    if (topics == None):
        topics = GetAllTopics(json_file)

    grouped_json = {}
    for topic in topics:
        grouped_json[topic] = []
        for i in json_file:
            if (i["topic"] == topic):
                grouped_json[topic].append(i)

    for topic in grouped_json:
        grouped_json[topic].sort(key=lambda x: x["name"])

    return grouped_json


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", type=str,
                        required=True, help="Readme file")
    parser.add_argument("--json_dump", type=str,
                        required=True, help="JSON dump file")
    parser.add_argument("--group_by_datatype",
                        action="store_true", help="Group by data type")
    return parser.parse_args()


def MakeElement(json_obj: json) -> str:
    element = "1. ![image][{}]".format(json_obj["datatype"])

    if (json_obj["url"] != None):
        element += " [{}]({})".format(json_obj["name"], json_obj["url"])
    else:
        element += " {}".format(json_obj["name"], json_obj["url"])

    if (json_obj["authors"]):
        element += " - {}".format(json_obj["authors"])
    element += "\n"

    return element


def main():
    args = ParseArgs()

    json_file = json.load(open(args.json_dump, "r"))
    topics = GetAllTopics(json_file)
    grouped_json = GroupByTopics(json_file, topics)

    with open(args.readme, "w") as f:
        f.write(METADATA + "\n\n")
        for shield in SHIELD_VARIABLES:
            f.write(SHIELD_VARIABLES[shield] + "\n")
        f.write("\n")

        f.write("### Table of contents\n")
        for topic in topics:
            f.write("- [{}](#{})\n".format(topic, NormalizeTopic(topic)))
        f.write("\n")

        for topic in topics:
            f.write("# {}\n".format(topic))

            if (args.group_by_datatype):
                for datatype in SHIELD_VARIABLES:
                    for i in grouped_json[topic]:
                        if (i["datatype"] == datatype):
                            f.write(MakeElement(i))
            else:
                for i in grouped_json[topic]:
                    f.write(MakeElement(i))
            f.write("\n")


if (__name__ == "__main__"):
    main()
