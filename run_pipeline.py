
import os
import logging

import tensorflow as tf
import apache_beam as beam
from apache_beam.options.pipeline_options import SetupOptions

from pipelines import process_docs
import parameters

from datetime import datetime


def main():

    tf.logging.set_verbosity(tf.logging.ERROR)
    logging.getLogger().setLevel(logging.INFO)

    # remove transformation directories (if exist)
    try:
        tf.gfile.DeleteRecursively(parameters.TEMP_DIR)
        tf.gfile.DeleteRecursively(parameters.TRANSFORMED_DATA_DIR)
        tf.gfile.DeleteRecursively(parameters.TRANSFORM_ARTEFACTS_DIR)
    except:
        pass  # ignore errors

    print 'Transform directories are removed...'
    print ''

    runner = parameters.RUNNER

    job_name = 'process-reuters-docs-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S'))
    print 'Launching {} job {} ... hang on'.format(runner, job_name)
    print("")

    # set pipeline arguments
    args = {
        'region': 'europe-west1',
        'staging_location': os.path.join(parameters.TEMP_DIR, 'staging'),
        'temp_location': parameters.TEMP_DIR,
        'job_name': job_name,
        'project': parameters.GCP_PROJECT_ID,
        'worker_machine_type': 'n1-standard-1',
        'max_num_workers': 50,
        'setup_file': './setup.py',
    }

    pipeline_options = beam.pipeline.PipelineOptions(flags=[],**args)

    process_docs.run_transformation_pipeline(runner, pipeline_options)


if __name__ == '__main__':
    main()

