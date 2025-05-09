import argparse

from utils.data_utils import merge_df_responses


def main(df_path: str, df_main_path):
    merge_df_responses(df_path=df_main_path, main_df_path=df_main_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--df_path', type=str, required=True, help="path to the model with pref results")
    parser.add_argument('--df_main_path', type=str, required=True, help="path to the model with no pref results")
    args = parser.parse_args()

    main(args.df_path, args.df_main_path)