import argparse
import json


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dump", type=str,
                        required=True, help="JSON dump file")
    return parser.parse_args()


def main():
    args = ParseArgs()

    json_file = json.load(open(args.json_dump, "r"))
    json_file.sort(key=lambda x: x["name"])
    json.dump(json_file, open(args.json_dump, "w"), indent=4)


if (__name__ == "__main__"):
    main()
