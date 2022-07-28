import os

import numpy as np
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores, matches):
    scores = scores.detach().cpu().numpy()
    pos_scores = [s for (s, m) in zip(scores, matches) if m == 1]
    neg_scores = [s for (s, m) in zip(scores, matches) if m == 0]

    positive_avg, negative_avg = np.mean(pos_scores), np.mean(neg_scores)
    if len(pos_scores) == 0:
        positive_avg = 0.0

    # positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx, save=False):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    if batch_idx % 2000 == 0 or save:
        name = os.path.join(path, "colbert.dnn")
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)

    if batch_idx in SAVED_CHECKPOINTS:
        name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
