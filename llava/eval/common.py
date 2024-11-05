import numpy as np

from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


def convert_none_string(value):
    """Convert the string 'None' to the actual None type."""
    if value == "None":
        return None
    return value


def get_input_ids(question, tokenizer, conv_mode, mm_use_im_start_end=False, image=True):
    """Given a question, creates prompt by adding image token and role, and returns tokenized prompt."""
    if image:
        # Add image token
        if mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_IMAGE_TOKEN + '\n' + question
    # Add role and format according to conversation template
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    return input_ids


def compute_n_gram_scores(references, hypothesis, verbose=False):
    """Compute n-gram-based metrics given a list of reference texts and a hypothesis text."""
    # Compute average ROUGE-1 score (F1-score based on overlap of 1-grams)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_score = np.mean([scorer.score(ref, hypothesis)['rouge1'].fmeasure for ref in references])

    references = [reference.split() for reference in references]
    hypothesis = hypothesis.split()

    # Calculate METEOR score (based on harmonic mean of 1-gram precision and recall, with recall weighted higher than precision)
    # With multiple references, the best score is chosen.
    m_score = meteor_score(references, hypothesis)

    # Calculate BLEU-1 score (1-gram precision)
    # This is the average of BLEU scores computed for each reference.
    bleu_1_score = sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0))

    if verbose:
        print(f"METEOR score: {m_score:.4f}")
        print(f"BLEU-1 score: {bleu_1_score:.4f}")
        print(f"ROUGE score: {rouge_score:.4f}")
    return m_score, bleu_1_score, rouge_score
