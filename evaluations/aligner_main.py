import argparse

from enums import PrefType
from evaluations.aligner import run_aligner


def main(df_path:str, model_path: str, pref_type:str):
    run_aligner(df_path, model_path, pref_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--df_path', type=str, required=True, help="data path having nopref_res")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    parser.add_argument('--pref_type', type=str, choices=[e.value for e in PrefType], required=True, help="whether relevant or irrevant")

    args = parser.parse_args()

    main(args.df_path, args.model_path, args.pref_type)