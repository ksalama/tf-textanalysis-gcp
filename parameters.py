import os


MODULE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/1'
#MODULE_URL = 'https://tfhub.dev/google/nnlm-en-dim50/1'

GCP_PROJECT_ID = 'ksalama-gcp-playground'
BQ_DATASET = 'playground_ds'
BQ_TABLE = 'reuters_embeddings'

ROOT = 'gs://cloudml-textanalysis'  # 'gs://cloudml-textanalysis' | '.'

DATA_DIR = os.path.join(ROOT, 'data/reuters')
TRANSFORMED_DATA_DIR = os.path.join(DATA_DIR, 'transformed')


RAW_TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
RAW_EVAL_DATA_DIR = os.path.join(DATA_DIR, 'eval')

TRANSFORMED_TRAIN_DATA_FILE_PREFIX = os.path.join(TRANSFORMED_DATA_DIR, 'train/docs-encoding')
TRANSFORMED_EVAL_DATA_FILE_PREFIX = os.path.join(TRANSFORMED_DATA_DIR, 'eval/docs-encoding')
TRANSFORMED_ENCODING = 'text'

TEMP_DIR = os.path.join(DATA_DIR, 'tmp')

MODELS_DIR = os.path.join(ROOT, 'models/reuters')

TRANSFORM_ARTEFACTS_DIR = os.path.join(MODELS_DIR,'transform')

RUNNER = 'DataflowRunner'  # DirectRunner | DataflowRunner

TEST_MODE = True  # process only few docs for testing

TEST_MODE_SAMPLE = 4  # number of topics & number of docs in each topic to process

DELIMITERS = '.,!?() '
VOCAB_SIZE = 50000