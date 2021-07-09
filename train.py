import numpy as np
import time
import os

import torch
from torch import nn

from models.model_factory import *

from dataset.dataset_factory import Covid_CT
from dataset.dataset_utils import prepare_torch_dataset

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.metrics import auc, roc_curve, average_precision_score, plot_roc_curve


ARGS = {
    "dataset_path": "./",
    "batch_size":   32,
    "augmentation": False,
    "test_path":    "./experiments/",
    "model_args": {
        "checkpoint_dir":  "./checkpoints/",
        "load_checkpoint": False,
        "is_cuda":         True,
        "apply_mixup":     True,
        "model_type":      "resnet50",
        "model_params":    {
            "num_classes":   2,
            "feature_dim":   512,
            "attention_dim": 256,
            "backbone":      "resnet50",
            "pretrain_path": None
        },
        "epochs":          100,
        "lr":              2e-5,
    }
}


train_dataset = Covid_CT(ARGS["dataset_path"], name="COVID-CT",
                         split='train',
                         from_numpy=True,
                         shuffle=True,
                         augmentation=ARGS["augmentation"])

valid_dataset = Covid_CT(ARGS["dataset_path"], name="COVID-CT",
                         split='valid',
                         from_numpy=True,
                         shuffle=False)

test_dataset = Covid_CT(ARGS["dataset_path"], name="COVID-CT",
                        split='test',
                        from_numpy=True,
                        shuffle=False)


TRAIN_DATASET = prepare_torch_dataset(train_dataset, ARGS["batch_size"], shuffle=True)
VALID_DATASET = prepare_torch_dataset(valid_dataset, ARGS["batch_size"], shuffle=False)
TEST_DATASET  = prepare_torch_dataset(test_dataset,  ARGS["batch_size"], shuffle=False)


model = Model(ARGS["model_args"])



def test_model(model, test_ds):
    true_labels = np.zeros((1, 203))
    predictions = np.zeros((1, 203))

    model.eval()
    with torch.no_grad():
        for n, (inp, tar) in enumerate(test_ds):
            inp = inp.cuda()
            # out = best_model(input).cpu().numpy()
            out = model(inp).cpu().numpy()
            # out = out.cpu().numpy()

            predictions[0, n*ARGS["batch_size"]:(n+1)*ARGS["batch_size"]] = np.argmax(out, axis=-1).reshape(1, -1)
            true_labels[0, n*ARGS["batch_size"]:(n+1)*ARGS["batch_size"]] = np.argmax(tar.numpy(), axis=-1).reshape(1, -1)

    true_labels = true_labels[0]
    predictions = predictions[0]

    log = str(dict(ARGS))
    log += "\n\n" + "-"*50 + "\n"

    cnfs = confusion_matrix(true_labels, predictions)
    tn = cnfs[0,0]; tp = cnfs[1,1]
    fn = cnfs[1,0]; fp = cnfs[0,1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bacc = (sensitivity + specificity) / 2.
    ppcr = (tp + fp) / (tp + fp + tn + fn)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    log +=  classification_report(true_labels, predictions)
    log += "\n\n>> Confusion Matrix:\n" + str(cnfs)
    log += "\n\n>> Scores:\n"
    log += f":: Accuracy:    {(tp+tn) / (tp+tn+fp+fn)}\n"
    log += f":: f1-score:    {2 * (precision*recall) / (precision+recall)}\n"
    log += f":: AUC:         {auc(fpr, tpr)}\n\n"
    log += f":: ROC AUC:     {roc_auc_score(true_labels, predictions)}\n"
    log += f":: Avr Prec:    {average_precision_score(true_labels, predictions)}\n\n"
    log += f":: Sensitivity: {sensitivity}\n"
    log += f":: Specificity: {specificity}\n"
    log += f":: bACC:        {bacc}\n"
    log += f":: PPCR:        {ppcr}\n"

    print(log)

    file_name = "{}{}_pre-{}_{:.0f}.txt"
    file_name = file_name.format(ARGS["test_path"],
                                 ARGS["model_args"]["model_type"],
                                 ARGS["model_args"]["model_params"]["pretrain_path"],
                                 time.time())

    print(file_name)
    with open(file_name, "w") as file:
        file.write(log)


if __name__ == "__main__":
    best_model = model.train(TRAIN_DATASET, VALID_DATASET)
    test_model(best_model, TEST_DATASET)
