import argparse

from enums import PrefType, PromptMethod
from evaluations.mcq_task_1 import full_pref_qa_task


def main(model_path: str, pref_type: str, prompt_method: str, enable_thinking:int,
         chunk: int = None, chunk_size: int = None):
    to_think = False
    if enable_thinking == 1:
        to_think = True
    print(f"Printing thinking here {to_think}")
    # if (pref_type == PrefType.IRRELEVANT_SET.value) or ("70" in model_path):
    #     print("running full")
    full_pref_qa_task(model_path, pref_type, prompt_method, to_think, chunk, chunk_size )
    # # else:
    # full_qa_task(model_path, pref_type, prompt_method, enable_thinking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Main Job")
    parser.add_argument('--model_path', type=str, required=True, help="mcq_model_path from huggingface")
    parser.add_argument('--pref_type', type=str, choices=[e.value for e in PrefType], required=True, help="whether relevant or irrevant")
    parser.add_argument('--prompt_method', type=str, choices=[e.value for e in PromptMethod], required=True, help="whether icl, cot or direct")
    parser.add_argument('--enable_thinking', type=int, required=True, help="1 if thinking is enabled, else 0")

    # Optional arguments
    parser.add_argument('--chunk', type=int, default=None, help="Which chunk to process")
    parser.add_argument('--chunk_size', type=int, default=None, help="total chunks")

    args = parser.parse_args()

    main(args.model_path, args.pref_type, args.prompt_method, args.enable_thinking,
         chunk=args.chunk, chunk_size=args.chunk_size)