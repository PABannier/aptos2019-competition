import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.id_code.values
    y = df.diagnosis.values

    skf = StratifiedKFold(n_splits=5)

    for fold, (trn_, val_)  in enumerate(skf.split(X, y)):
        df.loc[val_, "kfold"] = fold
    
    print(df.kfold.value_counts())

    df.to_csv("../input/train_fold.csv", index=False)