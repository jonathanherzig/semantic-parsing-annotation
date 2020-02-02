import argparse
import datetime
import json
import logging
import os
import random

from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
from torch import cuda


def _get_logger():
    DIR = os.path.dirname(__file__)
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(DIR, '../logs/log_{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('test_den_acc', 'dev_seq_acc', 'test_seq_acc',
                                                                'train_loss', 'best_val_loss', 'best_epoch',
                                                                'batch_size', 'lr', 'do'))
    return logger


def _run_experiment(config_file, serialization_dir, config_override, embeddings, cuda_ind, domain, learning_rate,
                    dropout):
    config_override["trainer"] = {"optimizer": {"lr": learning_rate}, "cuda_device": cuda_ind}
    or_model = {}
    if embeddings == 'elmo':
        or_model["source_embedder"] = {"elmo": {"dropout": dropout}}
    if domain is not None:
        or_model["domain"] = domain
    config_override["model"] = or_model

    train_model_from_file(parameter_filename=config_file,
                          serialization_dir=serialization_dir,
                          overrides=json.dumps(config_override),
                          force=True)


def _run_all(config_file, serialization_dir, scores_dir, config_override, embeddings, cuda_ind, domain, learning_rates,
             dropouts):
    """Runs an experiment for each hyperparams configuration and logs all results."""
    for learning_rate in learning_rates:
        for dropout in dropouts:
            _run_experiment(config_file, serialization_dir, config_override, embeddings, cuda_ind, domain,
                            learning_rate, dropout)

            score = json.load(open(scores_dir, 'r'))
            test_den_acc = score['test_den_acc']
            dev_seq_acc = score['best_validation_seq_acc']
            test_seq_acc = score['test_seq_acc']
            best_epoch = score['best_epoch']
            train_loss = score['training_loss']
            best_validation_loss = score['best_validation_loss']
            logger.info('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(test_den_acc, dev_seq_acc,
                                                                    test_seq_acc, train_loss, best_validation_loss,
                                                                    best_epoch, learning_rate, dropout))


def _parse_args():
    parser = argparse.ArgumentParser(
      description='experiment parser.',
      formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--embeddings', '-e', default='glove', choices=['glove', 'elmo'],
                        help='Pretrained embeddings to use (options: [glove, elmo]).')
    parser.add_argument('--version', '-v', default='nat', choices=['nat', 'lang', 'granno', 'overnight'],
                        help='Training version to use (options: [nat, lang, granno, overnight]).')
    parser.add_argument('--domain', '-d', default='geo', choices=['geo', 'scholar'],
                        help='(options: [geo, scholar]).')
    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_args()
    embeddings = args.embeddings
    domain = args.domain
    version = args.version

    serialization_dir = "tmp/output"
    scores_dir = "tmp/output/metrics.json"
    if embeddings == 'elmo':
        config_file = "configs/config_elmo.json"
    else:
        config_file = "configs/config_glove.json"

    num_devices_available = cuda.device_count()
    print('num_devices_available={}'.format(num_devices_available))
    config_override = dict()
    cuda_ind = 0 if num_devices_available > 0 else -1  # train on gpu, if possible

    config_override["train_data_path"] = "data/train_{}_{}.json".format(domain, version)
    config_override["validation_data_path"] = "data/dev_{}_{}.json".format(domain, version)
    config_override["test_data_path"] = "data/test_{}.json".format(domain)

    random.seed(0)
    import_submodules('nsp')

    logger = _get_logger()

    # hyper-params to search over
    if embeddings == 'elmo':
        dropouts = [0.4, 0.0, 0.1, 0.2, 0.3, 0.5]
    else:
        dropouts = [0.0]
    learning_rates = [0.01, 0.007, 0.013]

    _run_all(config_file, serialization_dir, scores_dir, config_override, embeddings, cuda_ind, domain, learning_rates,
             dropouts)
