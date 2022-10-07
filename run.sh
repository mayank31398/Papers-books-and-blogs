python scripts/make_readme.py --readme README.md --json_dump json_dump.json --group_by_datatype --add_shields
python scripts/sort_json.py --json_dump json_dump.json
pre-commit run --all-files
