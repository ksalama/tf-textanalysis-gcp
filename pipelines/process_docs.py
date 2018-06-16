import os
import logging

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.coders as tft_coders

from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata

import parameters


def create_bigquey_schema():

    from apache_beam.io.gcp.internal.clients import bigquery

    table_schema = bigquery.TableSchema()

    topic_schema = bigquery.TableFieldSchema()
    topic_schema.name = 'topic'
    topic_schema.type = 'string'
    topic_schema.mode = 'nullable'
    table_schema.fields.append(topic_schema)

    title_schema = bigquery.TableFieldSchema()
    title_schema.name = 'title'
    title_schema.type = 'string'
    title_schema.mode = 'nullable'
    table_schema.fields.append(title_schema)

    embed_schema = bigquery.TableFieldSchema()
    embed_schema.name = 'embeddings'
    embed_schema.type = 'float'
    embed_schema.mode = 'repeated'
    table_schema.fields.append(embed_schema)

    return table_schema


def get_paths(directory):

    import tensorflow as tf
    sub_directories = tf.gfile.ListDirectory(directory)

    if parameters.TEST_MODE:
        sub_directories = sub_directories[:parameters.TEST_MODE_SAMPLE]

    return [os.path.join(directory, path) for path in sub_directories if '.DS_' not in path]


def get_name_and_content(file_name):

    import tensorflow as tf
    content = tf.gfile.GFile(file_name).read()
    return file_name, content


def get_title_and_topic((file_name, content)):

    topic = file_name.split('/')[-2]
    title = content.split('\r')[1].replace('\n', '')
    return topic, title


def clean_text(text):

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk

    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        logging.warning('nltk resources exists')
    except LookupError:
        logging.info('Downloading nltk resources')
        nltk.download('punkt')
        nltk.download('stopwords')
        logging.info('nltk resources were downloaded')

    stop_words = stopwords.words('english')
    tokenized_words = word_tokenize(text.lower())
    tokenized_words = [''.join(c for c in word if c.isalnum()) for word in tokenized_words]
    clean_text = ' '.join([word.strip() for word in tokenized_words if word not in stop_words and word != ''])
    return clean_text


def to_dictionary(input_tuple):

    output_dict = dict()
    output_dict['topic'] = input_tuple[0]
    output_dict['raw_title'] = input_tuple[1]
    output_dict['clean_title'] = input_tuple[2]
    return output_dict


def get_raw_metadata():

    raw_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        'topic': dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'raw_title': dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'clean_title': dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation()),
    }))

    return raw_metadata


def get_embeddings(text):

    import tensorflow_hub as hub
    embed = hub.Module(parameters.MODULE_URL)
    embeddings = embed(text)
    return embeddings


def preprocessing_fn(input_features):

    # get the text of clean_title
    text = input_features['clean_title']

    # extract embeddings using tf.hub
    embeddings = tft.apply_function(get_embeddings, text)

    # tokenize text
    text_tokens = tf.string_split(text, parameters.DELIMITERS)

    # bag of words (bow) indices
    text_tokens_indices = tft.string_to_int(text_tokens, top_k=parameters.VOCAB_SIZE)

    # tf.idf
    bag_of_words_indices, tf_idf = tft.tfidf(text_tokens_indices, parameters.VOCAB_SIZE + 1)

    output_features = dict()
    output_features['topic'] = input_features['topic']
    output_features['title'] = input_features['raw_title']
    output_features['bow'] = bag_of_words_indices
    output_features['tf_idf'] = tf_idf
    output_features['embeddings'] = embeddings

    return output_features


def to_bq_row(entry):

    valid_embeddings = [round(float(e), 3) for e in entry['embeddings']]

    return {
        "topic": entry['topic'],
        "title": entry['title'],
        "embeddings": valid_embeddings
    }


############ Beam Pipeline Functions ####################################

def read_raw_data(pipeline, source, step):
    raw_data = (
            pipeline
            | '{} - Get Directories'.format(step) >> beam.Create(get_paths(source))
            | '{} - Get Files'.format(step) >> beam.FlatMap(get_paths)
            | '{} - Read Content'.format(step) >> beam.Map(get_name_and_content)
            | '{} - Get Title & Topic'.format(step) >> beam.Map(get_title_and_topic)
            | '{} - Clean Title'.format(step) >> beam.Map(lambda (topic, title): (topic, title, clean_text(title)))
            | '{} - To Dictionary'.format(step) >> beam.Map(to_dictionary)
    )

    return raw_data


def write_to_files(dataset, sink, encoding, step):

    data, metadata = dataset

    if encoding == 'tfrecords':
        (
                data
                | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=sink,
            file_name_suffix=".tfrecords",
            coder=tft_coders.example_proto_coder.ExampleProtoCoder(metadata.schema))
        )

    else:
        (
                data
                | '{} - Write Transformed Data'.format(step) >> beam.io.textio.WriteToText(
            file_path_prefix=sink,
            file_name_suffix=".txt")
        )


def write_to_bq(dataset, project_id, bq_dataset_name, bq_table_name, bq_table_schema, step):

    data, metadata = dataset

    (
            data
            | '{} - Convert to Valid BQ Row'.format(step) >> beam.Map(to_bq_row)
            | '{} - Write to BigQuery'.format(step) >> beam.io.WriteToBigQuery(
        project=project_id, dataset=bq_dataset_name, table=bq_table_name, schema=bq_table_schema,
        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
        write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
        )
    )


#####################################################################################################

def run_transformation_pipeline(runner, pipeline_options):

    with beam.Pipeline(runner, options=pipeline_options) as pipeline:
        with impl.Context(parameters.TEMP_DIR):

            ################## train data ##################

            step = 'Train'

            ### read raw train data
            raw_train_data = read_raw_data(pipeline, parameters.RAW_TRAIN_DATA_DIR, step)

            ### create a dataset from the train data and schema
            raw_train_dataset = (raw_train_data, get_raw_metadata())

            ### analyze and transform raw_train_dataset to produced transformed_dataset and transform_fn
            transformed_train_dataset, transform_fn = (
                    raw_train_dataset
                    | 'Analyze & Transform' >> impl.AnalyzeAndTransformDataset(
                preprocessing_fn)
            )

            ### write transformed train data to files
            write_to_files(
                transformed_train_dataset,
                parameters.TRANSFORMED_TRAIN_DATA_FILE_PREFIX,
                parameters.TRANSFORMED_ENCODING,
                step,
            )

            ### write to train embeddings to BigQuery
            write_to_bq(
                transformed_train_dataset,
                parameters.GCP_PROJECT_ID,
                parameters.BQ_DATASET,
                parameters.BQ_TABLE,
                create_bigquey_schema(),
                step
            )

            ################## eval data ##################

            step = 'Eval'

            ### read raw eval data
            raw_eval_data = read_raw_data(pipeline, parameters.RAW_EVAL_DATA_DIR, step)

            ### create a dataset from the train data and schema
            raw_eval_dataset = (raw_eval_data, get_raw_metadata())

            ### transform eval data based on produced transform_fn (from analyzing train_data)
            transformed_eval_dataset = (
                (raw_eval_dataset, transform_fn)
                | '{} - Transform'.format(step) >> impl.TransformDataset()
            )

            ### write transformed eval data to files
            write_to_files(
                transformed_train_dataset,
                parameters.TRANSFORMED_EVAL_DATA_FILE_PREFIX,
                parameters.TRANSFORMED_ENCODING,
                step,
            )

            ### write to eval embeddings to BigQuery
            write_to_bq(
                transformed_eval_dataset,
                parameters.GCP_PROJECT_ID,
                parameters.BQ_DATASET,
                parameters.BQ_TABLE,
                create_bigquey_schema(),
                step
            )

        ################## write transformation artefacts ##################

            (
                transform_fn
                | 'Write Transform Artefacts' >> transform_fn_io.WriteTransformFn(
                parameters.TRANSFORM_ARTEFACTS_DIR
                )
            )
