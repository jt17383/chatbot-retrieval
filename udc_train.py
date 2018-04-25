import os
import datetime
import logging
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)

tf.flags.DEFINE_string("input_dir", "./data", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", "./runs", "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", tf.logging.DEBUG, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None , "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("num_steps", 20000, "train steps")
tf.flags.DEFINE_integer("eval_every", 1000, "Evaluate after this many train steps")

FLAGS = tf.flags.FLAGS
FLAGS.input_dir = os.path.expanduser(FLAGS.input_dir)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M")
MODEL_DIR = os.path.join(os.path.expanduser(FLAGS.model_dir), TIMESTAMP)
#MODEL_DIR = os.path.expanduser(FLAGS.model_dir)
TRAIN_FILE = os.path.join(FLAGS.input_dir, "train.tfrecords")
VALIDATION_FILE = os.path.join(FLAGS.input_dir, "valid.tfrecords")

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):
  hparams = udc_hparams.create_hparams()

  model_fn = udc_model.create_model_fn(
    hparams,
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR)

  input_fn_train = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)

  input_fn_eval = udc_inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)

  eval_metrics = udc_metrics.create_evaluation_metrics()
  
  eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

  dbg_hook = tfdbg.LocalCLIDebugHook()
  estimator.fit(
    input_fn=input_fn_train,
    steps=FLAGS.num_steps,
    monitors=[eval_monitor]
  )

if __name__ == "__main__":
  tf.app.run()
