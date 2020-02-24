import os
import ast
from tqdm import tqdm
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliAIDataset

import torch
import torch.nn as nn


DEVICE = "cuda"
TRAIN_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")

IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))
EPOCHS = int(os.environ.get("EPOCHS"))

TRAINING_FOLDS = ast.literal_eval(os.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.get("VALIDATION_FOLDS"))

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN"))
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

BASE_MODEL = os.environ.get("BASE_MODEL")

LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))


def loss_fn(out, target):
    return nn.MSELoss(out, target)

def train(model, dataset, data_loader, optimizer):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        image = d['image']
        diagnosis = d['diagnosis']

        image = image.to_device(DEVICE, type=torch.float)
        diagnosis = diagnosis.to_device(DEVICE, type=torch.float)

        optimizer.zero_grad()
        output = model(image)

        loss = loss_fn(output, diagnosis)

        loss.backward()
        optimizer.step()

def evaluate(model, dataset, data_loader):
    model.eval()

    final_loss = 0
    counter = 0

    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)):
        counter += 1
        image = d['image']
        image = d['diagnosis']

        image = image.to_device(DEVICE, type=torch.float)
        diagnosis = diagnosis.to_device(DEVICE, type=torch.float)

        output = model(image)

        loss = loss_fn(output, diagnosis)
        final_loss += loss
    
    return final_loss / counter



def main():

    try:
        model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    except KeyError:
        raise NotImplementedError

    model.to_device(DEVICE)

    train_dataset = BengaliAIDataset(
        folds=TRAINING_FOLDS, 
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        training=True,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )

    valid_dataset = BengaliAIDataset(
        folds=VALIDATION_FOLDS, 
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        training=False,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, 
        batch_size=TEST_BATCH_SIZE, 
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Pay attention: some schedulers need to step after every batch OR after every epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, 
                                                            factor=0.3, verbose=True)

    # Other ideas: early stopping to prevent overfitting

    for epoch in range(EPOCHS):
        train(train_dataset, train_loader, model, optimizer)
        val_score = evaluate(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.h5')


if __name__ == "__main__":
    main()