import numpy as np
import string
from rouge_score import rouge_scorer, scoring
import collections
import functools
import re
from statistics import harmonic_mean

def normalize_answer(text):
  """QA style answer normalization. Similar to TriviaQA."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(string.punctuation)
    return "".join(" " if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def strip_attribution_tokens(text):
  """Strip the attribution tokens from an answer."""
  return re.sub(r'\[ ([1-9]) ([^\[\]]*) \]',r'\2' , text)


def non_quoted(text):
  """Returns only the text that is outside of quoted spans."""
  return re.sub(r'\[ ([1-9]) ([^\[\]]*) \]', '' , text)


def only_quoted(text, sources='1-9', sep = ' '):
  """Returns only the text that is within of quoted spans."""
  return sep.join([x.group(1) for x in re.finditer(r'\[ [{}] ([^\[\]]*) \]'.format(sources), text)])


def quoted_sources(text):
  """Returns the list of input sources that were quoted in the answer."""
  return sorted(list(set([int(x.group(1)) for x in re.finditer(r'\[ ([1-9]) [^\[\]]* \]', text)])))


def score_all(all_targets, all_predictions, scorer, aggr_measure, score_keys, preprocess_func=None, bootstrap=False):
  """
  Aggregates across all targets per sample.

  all_targets: list of list of strings
  all_predictions: list of strings
  """
  np.random.seed(1337)

  is_rouge_measure = 'rouge' in aggr_measure

  if preprocess_func is not None:
    scoring_func = lambda target, prediction: scorer.score(target=preprocess_func(target), prediction=preprocess_func(prediction))
  else:
    scoring_func = scorer.score

  aggregator = scoring.BootstrapAggregator()
  all_scores = [] if is_rouge_measure else dict((k,[]) for k in score_keys)
  for targets, prediction in zip(all_targets, all_predictions):
    # Max across references by aggr_measure
    if is_rouge_measure:
      max_scores = max([scoring_func(target, prediction) for target in targets], key=lambda x: x[aggr_measure].fmeasure)

      aggregator.add_scores(max_scores)
      all_scores.append(max_scores[aggr_measure].fmeasure*100)
    else:
      if aggr_measure == 'independent':
        max_scores = {}
        for key in score_keys:
          max_scores[key] = max([scoring_func(target, prediction)[key] for target in targets])
      else:
        max_scores = max([scoring_func(target, prediction) for target in targets], key=lambda x: x[aggr_measure])

      aggregator.add_scores(max_scores)
      for key in score_keys:
        all_scores[key].append(max_scores[key]*100)

  if not bootstrap:
    return all_scores

  result = aggregator.aggregate()
  postprocess_result = (lambda x: x.fmeasure*100) if is_rouge_measure else (lambda x: x*100)
  bootstrap_results = {}
  for key in score_keys:
    bootstrap_results[key] = (postprocess_result(result[key].mid), postprocess_result(result[key].low), postprocess_result(result[key].high))
  return bootstrap_results, all_scores

## ROUGE ##

score_all_rouge = functools.partial(score_all, scorer=rouge_scorer.RougeScorer(rouge_types=("rouge1", "rouge2", "rougeLsum", "rougeL")), aggr_measure='rougeLsum',  score_keys=("rouge1", "rouge2", "rougeLsum"), preprocess_func=strip_attribution_tokens)

## F1 ##

class _f1_scorer:
  def score(self, target, prediction):
    """Computes token F1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(target_tokens))
    num_same = sum(common.values())
    if len(target_tokens) == 0 and len(prediction_tokens) == 0:
      return {'F1': 1.0, 'recall': 1.0, 'precision': 1.0}
    elif len(target_tokens) == 0 and len(prediction_tokens) > 0:
      return {'F1': 0.0, 'recall': 1.0, 'precision': 0.0}
    elif len(target_tokens) > 0 and len(prediction_tokens) == 0:
      return {'F1': 0.0, 'recall': 0.0, 'precision': 1.0}
    elif num_same == 0:
      return {'F1': 0.0, 'recall': 0.0, 'precision': 0.0}
    else:
      precision = 1.0 * num_same / len(prediction_tokens)
      recall = 1.0 * num_same / len(target_tokens)
      f1 = (2 * precision * recall) / (precision + recall)
      return {'F1': f1, 'recall': recall, 'precision': precision}


score_all_f1 = functools.partial(score_all, scorer=_f1_scorer(), aggr_measure='F1', score_keys=("F1", "recall", "precision"))


def preprocess_quotes_f1(text, sep=' ', sources='1-7'):
  text = only_quoted(text, sep=sep, sources=sources)
  return normalize_answer(text)


def score_semqa_f1(all_targets, all_predictions, examples, harmonic=False):
  per_source_prf1 = {}
  for source in range(1, 8):
    preprocess_quotes_f1_partial_sources = functools.partial(preprocess_quotes_f1, sep=' ', sources=f'{source}')
    scores = score_all_f1(all_targets, all_predictions, aggr_measure='independent', preprocess_func=preprocess_quotes_f1_partial_sources)

    for aggr_measure in ('F1', 'recall', 'precision'):
      per_source_prf1[f'{aggr_measure}_source_{source}'] = scores[aggr_measure]

  semqa_f1s = []
  for i in range(len(examples)):
    precisions, recalls, f1s = [], [] , []
    for source in range(1,8):
      if examples[i][0][f'source{source}']:
        precisions.append(per_source_prf1[f'precision_source_{source}'][i])
        recalls.append(per_source_prf1[f'recall_source_{source}'][i])
        f1s.append(per_source_prf1[f'F1_source_{source}'][i])
    if harmonic:
      f1 = harmonic_mean(precisions + recalls)
    else:
      f1 = np.mean(f1s)
    semqa_f1s.append(f1)

  return np.mean(semqa_f1s)


score_all_recall = functools.partial(score_all, scorer=_f1_scorer(), aggr_measure='recall', score_keys=("recall",))


def score_semqa_short_recall(all_targets, all_predictions):

  # Ignore examples with no targets.
  non_empty_targets, non_empty_predictions = [], []
  for tar, pred in zip(all_targets, all_predictions):
    if len(tar) == 0 or all([x == '' for x in tar]):
      continue
    non_empty_targets.append(tar)
    non_empty_predictions.append(pred)

  per_source_recall = {}
  for source in range(1, 8):
    preprocess_quotes_f1_partial_sources = functools.partial(preprocess_quotes_f1, sep=' ', sources=f'{source}')
    scores = score_all_recall(non_empty_targets, non_empty_predictions, preprocess_func=preprocess_quotes_f1_partial_sources)
    per_source_recall[f'recall_source_{source}'] = scores['recall']

  semqa_recalls = []
  for i in range(len(non_empty_targets)):
    recalls = []
    for source in range(1,8):
      preprocess_quotes_f1_partial_sources = functools.partial(preprocess_quotes_f1, sep=' ', sources=f'{source}')
      if any([preprocess_quotes_f1_partial_sources(tar) for tar in non_empty_targets[i]]):
        recalls.append(per_source_recall[f'recall_source_{source}'][i])
      avg_recalls = np.mean(recalls)
    semqa_recalls.append(avg_recalls)

  return np.mean(semqa_recalls)

