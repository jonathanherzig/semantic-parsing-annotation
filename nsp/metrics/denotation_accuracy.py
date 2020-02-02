from typing import List, Dict

from overrides import overrides

from allennlp.training.metrics import Metric

import tempfile
import subprocess
import os
import time


@Metric.register("denotation_accuracy")
class DenotationAccuracy(Metric):
    """
    Calculates the denotation accuracy.
    """

    def __init__(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._true_answers = []
        self._preds = []
        self._vars = []
        self._indices = []

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._true_answers = []
        self._preds = []
        self._indices = []

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]], variables: List[dict], indices: List[int], domain: str) -> None:
        self._total_counts += len(predictions)
        self._domain = domain
        self._vars.append(variables)
        for predicted_tokens, gold_tokens, var, index in zip(predictions, gold_targets, variables, indices):

            true_answer_rep = []
            for t in gold_tokens:
                next_token = lexicalize_entity(var, t)
                true_answer_rep.append(next_token)

            self._true_answers.append(" ".join(true_answer_rep))
            self._indices.append(index)

            preds = []
            for pred in predicted_tokens:

                pred_rep = []
                for t in pred:
                    next_token = lexicalize_entity(var, t)
                    pred_rep.append(next_token)

                preds.append(" ".join(pred_rep))
            self._preds.append(preds)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        accuracy = 0.

        all_lfs = ([format_lf(s) for s in self._true_answers] +
                   [format_lf(s) for p in self._preds for s in p])

        if reset and all_lfs:
            tf_lines = all_lfs
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".examples")
            for line in tf_lines:
                print(line, file=tf)
            tf.flush()
            FNULL = open(os.devnull, "w")
            if self._domain == "geo":
                evaluator_name = "evaluator/overnight"
                subdomain = "geo880"
            else:
                evaluator_name = "evaluator/scholar"
                subdomain = "external"
            start_time = time.time()
            msg = subprocess.check_output([evaluator_name, subdomain, tf.name], stderr=FNULL)
            print("execution took {0} seconds".format(time.time() - start_time))
            tf.close()

            denotations = [line.split("\t")[1] for line in msg.decode("utf-8").split("\n")
                           if line.startswith("targetValue\t")]
            true_dens = denotations[:len(self._true_answers)]
            all_pred_dens = denotations[len(self._true_answers):]

            all_pred_dens_ = []
            for i, pred_d in enumerate(all_pred_dens):
                all_pred_dens_.append(rep_to_empty_set(pred_d))
            all_pred_dens = all_pred_dens_
            true_dens = [rep_to_empty_set(pred_d) for pred_d in true_dens]

            derivs, pred_dens = pick_derivations(all_pred_dens, self._preds, is_error)
            match = [t == p for t, p in zip(true_dens, pred_dens)]

            ind_to_match = dict()
            for (i, m) in zip(self._indices, match):
                ind_to_match[i] = m

            accuracy = sum(match) / len(match)
            self.reset()

        return {"den_acc": accuracy}


def lexicalize_entity(var, token):
    """Lexicalize an abstract entity"""
    curr_year = 2016
    if token == "year0" and (token in var or "misc0" in var):
        if token in var:
            return "number {} year".format(var[token])
        else:
            return "number {} year".format(curr_year - int(var["misc0"]))
    elif token == "misc0" and token in var:
        return "number {} count".format(var[token])
    elif token in var:
        return var[token]
    else:
        return token


def format_lf(lf):
    replacements = [
        ("! ", "!"),
        ("SW", "edu.stanford.nlp.sempre.overnight.SimpleWorld"),
    ]
    for a, b in replacements:
        lf = lf.replace(a, b)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == "(")
    num_right_paren = sum(1 for c in lf if c == ")")
    diff = num_left_paren - num_right_paren
    if diff > 0:
        while len(lf) > 0 and lf[-1] == "(" and diff > 0:
            lf = lf[:-1]
            diff -= 1
        if len(lf) == 0: return ""
        lf = lf + " )" * diff
    return lf


def pick_derivations(all_pred_dens, all_derivs, is_error_fn):
    # Find the top-scoring derivation that executed without error
    derivs = []
    pred_dens = []
    cur_start = 0
    for deriv_set in all_derivs:
        for i in range(len(deriv_set)):
            cur_denotation = all_pred_dens[cur_start + i]
            if not is_error_fn(cur_denotation):
                derivs.append(deriv_set[i])
                pred_dens.append(cur_denotation)
                break
        else:
            derivs.append(deriv_set[0])  # Default to first derivation
            pred_dens.append(all_pred_dens[cur_start])
        cur_start += len(deriv_set)
    return (derivs, pred_dens)


def is_error(d):
    return "BADJAVA" in d or "ERROR" in d or d == "null"


def rep_to_empty_set(pred_den):
    """These execution errors indicate an empty set result"""
    if (pred_den == "BADJAVA: java.lang.RuntimeException: java.lang.NullPointerException" or pred_den == "BADJAVA: java.lang.RuntimeException: java.lang.RuntimeException: DB doesn't contain entity null"):
        return "(list)"
    else:
        return pred_den
