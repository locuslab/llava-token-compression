import argparse
import csv
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from llava.eval.common import convert_none_string, get_input_ids
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, get_model_name_from_path


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    references = pd.read_parquet(args.reference_file)
    references = references.values.tolist()
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    csvfile = open(answers_file, "w", newline='')
    fieldnames = ["question_id", "segment_id", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for _, line in enumerate(tqdm(references)):
        question_id, segment_id, images, question, _ = line
        image_file = images[-1] # Note: there are multiple images in a sequence for a given question. For now, use just first image.
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]
        input_ids = get_input_ids(
            question, tokenizer, args.conv_mode,
            mm_use_im_start_end=getattr(model.config, 'mm_use_im_start_end', False))
        input_ids = input_ids.unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        writer.writerow({"question_id": question_id, "segment_id": segment_id, "answer": outputs})
    csvfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/scratch/shared/models/huggingface/llava-v1.5-7b")
    parser.add_argument("--model-base", type=convert_none_string, default=None)
    parser.add_argument("--image-folder", type=str, default="/scratch/shared/datasets/lingoQA/evaluation/")
    parser.add_argument("--reference-file", type=str, default="/scratch/shared/datasets/lingoQA/evaluation/val.parquet")
    parser.add_argument("--answers-file", type=str, default="predictions.csv")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.)
    args = parser.parse_args()

    eval_model(args)
