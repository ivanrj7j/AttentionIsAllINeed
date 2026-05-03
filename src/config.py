import torch
from uuid import uuid4

MODEL_ID = str(uuid4())
# MODEL_ID = ""
# uncomment this to add custom model id 

EPOCHS = 5
STEPS_PER_EPOCH = 1
BATCH_SIZE = 64
SAVE_MODEL_EVERY = 2
# training info 

LEARNING_RATE = 1e-4
BETAS = (0.9, 0.98)
EPSILON = 1e-9
# optimizer info

PCT_START = 0.1
ANNEAL_STRATEGY = 'cos'
DIV_FACTOR = 10
FINAL_DIV_FACTOR = 1e4
# scheduler info 

SRC_VOCAB_SIZE = 4000
TGT_VOCAB_SIZE = 4000
EMBEDDING_SIZE = 512
HEAD_COUNT = 8
NUM_LAYERS = 6
DFF = 128
MAX_SEQ_LEN = 64
DROPOUT = 0.1
# model info 

TARGET_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# configuring the device 

SRC_FILE_TRAIN = '../dataset/engTrain.txt'
TGT_FILE_TRAIN = '../dataset/malTrain.txt'
SRC_FILE_TEST = '../dataset/engTest.txt'
TGT_FILE_TEST = '../dataset/malTest.txt'
SRC_FILE_VAL = '../dataset/engVal.txt'
TGT_FILE_VAL = '../dataset/malVal.txt'
MAX_SENTENCE_SIZE = 66
BUFFER_SIZE = 1000
TRAIN_WORKERS = 5
TEST_WORKERS, VALIDATION_WORKERS = 2, 2
# data loader info 

SRC_TOKENIZER = "../dataset/engTokenizer.json"
TGT_TOKENIZER = "../dataset/malTokenizer.json"
# tokenizers path

SUMMARY_PATH = f'../runs/{MODEL_ID}_transformer'