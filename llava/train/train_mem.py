import os
import sys
sys.path.append(os.getcwd())

from llava.train.train import train

os.environ["TF_DEVICE_MIN_SYS_MEMORY_IN_MB"] = "4096"

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
