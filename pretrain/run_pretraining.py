# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-trains an ELECTRA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os

import tensorflow.compat.v1 as tf

import configure_pretraining
from model import modeling
from model import optimization
from pretrain import pretrain_data
from pretrain import pretrain_helpers
from util import training_utils
from util import utils


class PretrainingModel(object):
    """Transformer pre-training using the replaced-token-detection task."""

    def __init__(
        self, config: configure_pretraining.PretrainingConfig, features, is_training
    ):
        # Set up model config
        self._config = config
        self._bert_config = training_utils.get_bert_config(config)
        if config.debug:
            self._bert_config.num_hidden_layers = 3
            self._bert_config.hidden_size = 144
            self._bert_config.intermediate_size = 144 * 4
            self._bert_config.num_attention_heads = 4

        # Mask the input
        masked_inputs = pretrain_helpers.mask(
            config, pretrain_data.features_to_inputs(features), config.mask_prob
        )

        # Generator
        embedding_size = (
            self._bert_config.hidden_size
            if config.embedding_size is None
            else config.embedding_size
        )
        if config.uniform_generator:
            mlm_output = self._get_masked_lm_output(masked_inputs, None)
        elif config.electra_objective and config.untied_generator:
            generator = self._build_transformer(
                masked_inputs,
                is_training,
                bert_config=get_generator_config(config, self._bert_config),
                embedding_size=(
                    None if config.untied_generator_embeddings else embedding_size
                ),
                untied_embeddings=config.untied_generator_embeddings,
                name="generator",
            )
            mlm_output = self._get_masked_lm_output(masked_inputs, generator)
        else:
            generator = self._build_transformer(
                masked_inputs, is_training, embedding_size=embedding_size
            )
            mlm_output = self._get_masked_lm_output(masked_inputs, generator)
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        self.mlm_output = mlm_output
        self.total_loss = config.gen_weight * mlm_output.loss

        # Discriminator
        disc_output = None
        if config.electra_objective:
            discriminator = self._build_transformer(
                fake_data.inputs,
                is_training,
                reuse=not config.untied_generator,
                embedding_size=embedding_size,
            )
            disc_output = self._get_discriminator_output(
                fake_data.inputs, discriminator, fake_data.is_fake_tokens
            )
            self.total_loss += config.disc_weight * disc_output.loss

        # Evaluation
        eval_fn_inputs = {
            "input_ids": masked_inputs.input_ids,
            "masked_lm_preds": mlm_output.preds,
            "mlm_loss": mlm_output.per_example_loss,
            "masked_lm_ids": masked_inputs.masked_lm_ids,
            "masked_lm_weights": masked_inputs.masked_lm_weights,
            "input_mask": masked_inputs.input_mask,
        }
        if config.electra_objective:
            eval_fn_inputs.update(
                {
                    "disc_loss": disc_output.per_example_loss,
                    "disc_labels": disc_output.labels,
                    "disc_probs": disc_output.probs,
                    "disc_preds": disc_output.preds,
                    "sampled_tokids": tf.argmax(
                        fake_data.sampled_tokens, -1, output_type=tf.int32
                    ),
                }
            )
        eval_fn_keys = eval_fn_inputs.keys()
        eval_fn_values = [eval_fn_inputs[k] for k in eval_fn_keys]

        def metric_fn(*args):
            """Computes the loss and accuracy of the model."""
            d = {k: arg for k, arg in zip(eval_fn_keys, args)}
            metrics = dict()
            metrics["masked_lm_accuracy"] = tf.metrics.accuracy(
                labels=tf.reshape(d["masked_lm_ids"], [-1]),
                predictions=tf.reshape(d["masked_lm_preds"], [-1]),
                weights=tf.reshape(d["masked_lm_weights"], [-1]),
            )
            metrics["masked_lm_loss"] = tf.metrics.mean(
                values=tf.reshape(d["mlm_loss"], [-1]),
                weights=tf.reshape(d["masked_lm_weights"], [-1]),
            )
            if config.electra_objective:
                metrics["sampled_masked_lm_accuracy"] = tf.metrics.accuracy(
                    labels=tf.reshape(d["masked_lm_ids"], [-1]),
                    predictions=tf.reshape(d["sampled_tokids"], [-1]),
                    weights=tf.reshape(d["masked_lm_weights"], [-1]),
                )
                if config.disc_weight > 0:
                    metrics["disc_loss"] = tf.metrics.mean(d["disc_loss"])
                    metrics["disc_auc"] = tf.metrics.auc(
                        d["disc_labels"] * d["input_mask"],
                        d["disc_probs"] * tf.cast(d["input_mask"], tf.float32),
                    )
                    metrics["disc_accuracy"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["input_mask"],
                    )
                    metrics["disc_precision"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["disc_preds"] * d["input_mask"],
                    )
                    metrics["disc_recall"] = tf.metrics.accuracy(
                        labels=d["disc_labels"],
                        predictions=d["disc_preds"],
                        weights=d["disc_labels"] * d["input_mask"],
                    )
            return metrics

        self.eval_metrics = (metric_fn, eval_fn_values)

    def _get_masked_lm_output(self, inputs: pretrain_data.Inputs, model):
        """Masked language modeling softmax layer."""
        masked_lm_weights = inputs.masked_lm_weights
        with tf.variable_scope("generator_predictions"):
            if self._config.uniform_generator:
                logits = tf.zeros(self._bert_config.vocab_size)
                logits_tiled = tf.zeros(
                    modeling.get_shape_list(inputs.masked_lm_ids)
                    + [self._bert_config.vocab_size]
                )
                logits_tiled += tf.reshape(logits, [1, 1, self._bert_config.vocab_size])
                logits = logits_tiled
            else:
                relevant_hidden = pretrain_helpers.gather_positions(
                    model.get_sequence_output(), inputs.masked_lm_positions
                )
                hidden = tf.layers.dense(
                    relevant_hidden,
                    units=modeling.get_shape_list(model.get_embedding_table())[-1],
                    activation=modeling.get_activation(self._bert_config.hidden_act),
                    kernel_initializer=modeling.create_initializer(
                        self._bert_config.initializer_range
                    ),
                )
                hidden = modeling.layer_norm(hidden)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[self._bert_config.vocab_size],
                    initializer=tf.zeros_initializer(),
                )
                logits = tf.matmul(
                    hidden, model.get_embedding_table(), transpose_b=True
                )
                logits = tf.nn.bias_add(logits, output_bias)

            oh_labels = tf.one_hot(
                inputs.masked_lm_ids,
                depth=self._bert_config.vocab_size,
                dtype=tf.float32,
            )

            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

            numerator = tf.reduce_sum(inputs.masked_lm_weights * label_log_probs)
            denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
            loss = numerator / denominator
            preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

            MLMOutput = collections.namedtuple(
                "MLMOutput", ["logits", "probs", "loss", "per_example_loss", "preds"]
            )
            return MLMOutput(
                logits=logits,
                probs=probs,
                per_example_loss=label_log_probs,
                loss=loss,
                preds=preds,
            )

    def _get_discriminator_output(self, inputs, discriminator, labels):
        """Discriminator binary classifier."""
        with tf.variable_scope("discriminator_predictions"):
            hidden = tf.layers.dense(
                discriminator.get_sequence_output(),
                units=self._bert_config.hidden_size,
                activation=modeling.get_activation(self._bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    self._bert_config.initializer_range
                ),
            )
            logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
            weights = tf.cast(inputs.input_mask, tf.float32)
            labelsf = tf.cast(labels, tf.float32)
            losses = (
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labelsf)
                * weights
            )
            per_example_loss = tf.reduce_sum(losses, axis=-1) / (
                1e-6 + tf.reduce_sum(weights, axis=-1)
            )
            loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
            probs = tf.nn.sigmoid(logits)
            preds = tf.cast(tf.round((tf.sign(logits) + 1) / 2), tf.int32)
            DiscOutput = collections.namedtuple(
                "DiscOutput", ["loss", "per_example_loss", "probs", "preds", "labels"]
            )
            return DiscOutput(
                loss=loss,
                per_example_loss=per_example_loss,
                probs=probs,
                preds=preds,
                labels=labels,
            )

    def _get_fake_data(self, inputs, mlm_logits):
        """Sample from the generator to create corrupted input."""
        inputs = pretrain_helpers.unmask(inputs)
        disallow = (
            tf.one_hot(
                inputs.masked_lm_ids,
                depth=self._bert_config.vocab_size,
                dtype=tf.float32,
            )
            if self._config.disallow_correct
            else None
        )
        sampled_tokens = tf.stop_gradient(
            pretrain_helpers.sample_from_softmax(
                mlm_logits / self._config.temperature, disallow=disallow
            )
        )
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = pretrain_helpers.scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions
        )
        labels = masked * (
            1 - tf.cast(tf.equal(updated_input_ids, inputs.input_ids), tf.int32)
        )
        updated_inputs = pretrain_data.get_updated_inputs(
            inputs, input_ids=updated_input_ids
        )
        FakedData = collections.namedtuple(
            "FakedData", ["inputs", "is_fake_tokens", "sampled_tokens"]
        )
        return FakedData(
            inputs=updated_inputs, is_fake_tokens=labels, sampled_tokens=sampled_tokens
        )

    def _build_transformer(
        self,
        inputs: pretrain_data.Inputs,
        is_training,
        bert_config=None,
        name="electra",
        reuse=False,
        **kwargs
    ):
        """Build a transformer encoder network."""
        if bert_config is None:
            bert_config = self._bert_config
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            return modeling.BertModel(
                bert_config=bert_config,
                is_training=is_training,
                input_ids=inputs.input_ids,
                input_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                use_one_hot_embeddings=self._config.use_tpu,
                scope=name,
                **kwargs
            )


def get_generator_config(
    config: configure_pretraining.PretrainingConfig, bert_config: modeling.BertConfig
):
    """Get model config for the generator network."""
    gen_config = modeling.BertConfig.from_dict(bert_config.to_dict())
    gen_config.hidden_size = int(
        round(bert_config.hidden_size * config.generator_hidden_size)
    )
    gen_config.num_hidden_layers = int(
        round(bert_config.num_hidden_layers * config.generator_layers)
    )
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config


def model_fn_builder(config: configure_pretraining.PretrainingConfig):
    """Build the model for training."""

    def model_fn(features, labels, mode, params):
        """Build the model for training."""
        model = PretrainingModel(config, features, mode == tf.estimator.ModeKeys.TRAIN)
        utils.log("Model is built!")
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                model.total_loss,
                config.learning_rate,
                config.num_train_steps,
                weight_decay_rate=config.weight_decay_rate,
                use_tpu=config.use_tpu,
                warmup_steps=config.num_warmup_steps,
                lr_decay_power=config.lr_decay_power,
                mixed_precision=config.mixed_precision,
            )
            if config.use_tpu:
                output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=model.total_loss,
                    train_op=train_op,
                    training_hooks=[
                        training_utils.ETAHook(
                            {} if config.use_tpu else dict(loss=model.total_loss),
                            config.num_train_steps,
                            config.iterations_per_loop,
                            config.use_tpu,
                        )
                    ],
                )
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=model.total_loss,
                    train_op=train_op,
                    training_hooks=[
                        training_utils.ETAHook(
                            {} if config.use_tpu else dict(loss=model.total_loss),
                            config.num_train_steps,
                            config.iterations_per_loop,
                            config.use_tpu,
                        )
                    ],
                )
        elif mode == tf.estimator.ModeKeys.EVAL:
            if config.use_tpu:
                output_spec = tf.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=model.total_loss,
                    eval_metrics=model.eval_metrics,
                    evaluation_hooks=[
                        training_utils.ETAHook(
                            {} if config.use_tpu else dict(loss=model.total_loss),
                            config.num_eval_steps,
                            config.iterations_per_loop,
                            config.use_tpu,
                            is_training=False,
                        )
                    ],
                )
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=model.total_loss,
                    eval_metrics=model.eval_metrics,
                    evaluation_hooks=[
                        training_utils.ETAHook(
                            {} if config.use_tpu else dict(loss=model.total_loss),
                            config.num_eval_steps,
                            config.iterations_per_loop,
                            config.use_tpu,
                            is_training=False,
                        )
                    ],
                )
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported")
        return output_spec

    return model_fn


def train_or_eval(config: configure_pretraining.PretrainingConfig):
    """Run pre-training or evaluate the pre-trained model."""
    if config.do_train == config.do_eval:
        raise ValueError("Exactly one of `do_train` or `do_eval` must be True.")
    if config.debug:
        utils.rmkdir(config.model_dir)
    utils.heading("Config:")
    utils.log_config(config)

    if config.use_tpu:
        is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
        tpu_cluster_resolver = None
        if config.use_tpu and config.tpu_name:
            tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                config.tpu_name, zone=config.tpu_zone, project=config.gcp_project
            )

        tpu_config = tf.estimator.tpu.TPUConfig(
            iterations_per_loop=config.iterations_per_loop,
            num_shards=config.num_tpu_cores,
            tpu_job_name=config.tpu_job_name,
            per_host_input_for_training=is_per_host,
        )
        run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=config.model_dir,
            save_checkpoints_steps=config.save_checkpoints_steps,
            tpu_config=tpu_config,
        )
        model_fn = model_fn_builder(config=config)
        estimator = tf.estimator.tpu.TPUEstimator(
            use_tpu=config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
        )
    else:
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(
            model_dir=config.model_dir,
            save_checkpoints_steps=config.save_checkpoints_steps,
            session_config=config_proto,
        )
        model_fn = model_fn_builder(config=config)

        estimator = None
        if config.saved_model:
            estimator = tf.estimator.Estimator(
                model_fn=model_fn, config=run_config, warm_start_from=config.saved_model
            )
        else:
            estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)

    if config.do_train:
        utils.heading("Running training")
        estimator.train(
            input_fn=pretrain_data.get_input_fn(config, True),
            max_steps=config.num_train_steps,
        )
    if config.do_eval:
        utils.heading("Running evaluation")
        result = estimator.evaluate(
            input_fn=pretrain_data.get_input_fn(config, False),
            steps=config.num_eval_steps,
        )
        for key in sorted(result.keys()):
            utils.log("  {:} = {:}".format(key, str(result[key])))
        return result


def train_one_step(config: configure_pretraining.PretrainingConfig):
    """Builds an ELECTRA model an trains it for one step; useful for debugging."""
    train_input_fn = pretrain_data.get_input_fn(config, True)
    features = tf.data.make_one_shot_iterator(
        train_input_fn(dict(batch_size=config.train_batch_size))
    ).get_next()
    model = PretrainingModel(config, features, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        utils.log(sess.run(model.total_loss))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="Location of data files (model weights, etc).",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="The name of the model being fine-tuned.",
    )
    parser.add_argument(
        "--hparams", default="{}", type=str, help="JSON dict of model hyperparameters."
    )
    parser.add_argument("--use_tpu", action="store_true", help="Using tpu.")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Using mixed precision."
    )
    args = parser.parse_args()

    if args.mixed_precision:
        os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
    if args.hparams.endswith(".json"):
        hparams = utils.load_json(args.hparams)
    else:
        hparams = json.loads(args.hparams)
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_or_eval(
        configure_pretraining.PretrainingConfig(
            args.model_name,
            args.data_dir,
            args.use_tpu,
            args.mixed_precision,
            **hparams
        )
    )


if __name__ == "__main__":
    main()
