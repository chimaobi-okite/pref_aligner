import argparse

from enums import PrefType, PromptMethod
from evaluations.full_mcq_task import full_qa_task
from evaluations.mcq_irr_task import full_pref_qa_task


def main(model_path: str, pref_type: str, prompt_method: str):
    if (pref_type == PrefType.IRRELEVANT_SET.value) or ("70" in model_path):
        print("running full")
        full_pref_qa_task(model_path, pref_type, prompt_method,)
    else:
        full_qa_task(model_path, pref_type, prompt_method,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path")
    parser.add_argument('--pref_type', type=str, choices=[e.value for e in PrefType], required=True, help="whether relevant or irrevant")
    parser.add_argument('--prompt_method', type=str, choices=[e.value for e in PromptMethod], required=True, help="whether icl, cot or direct")
    args = parser.parse_args()
    main(args.model_path, args.pref_type, args.prompt_method)