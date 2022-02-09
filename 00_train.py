import warnings

from utils import *
from DL_ClassifierModel import *
from progressbar import progressbar


warnings.filterwarnings("ignore")


CUDA_NUM = 2
EPOCHS = 100


def train_model(direction, seed=123456, cuda_num=CUDA_NUM):
    dataClass = DataClass_normal(direction)

    model = DTI_Bridge(seed=seed,
                       cSize=dataClass.pContFeat.shape[1],
                       device=torch.device(f"cuda:{cuda_num}"))

    return model.train(dataClass, seed, epochs=EPOCHS,
                       savePath=f"models/{direction}")



directions = [dir_.replace("_dir_dict.pkl", "")
              for dir_ in os.listdir("../get_data/Shallow/directions")]

training_rows = [",".join(["direction",
                           "AUC", "AUPR", "LogAUC", "recall_1", "recall_5",
                           "recall_10", "recall_25", "recall_50",
                          "EF_1", "EF_5", "EF_10", "EF_25" "EF_50"])]
validation_rows = [",".join(["direction",
                           "AUC", "AUPR", "LogAUC", "recall_1", "recall_5",
                           "recall_10", "recall_25", "recall_50",
                           "EF_1", "EF_5", "EF_10", "EF_25" "EF_50"])]
for direction in progressbar(directions):
    training_row, validation_row = train_model(direction)
    training_rows.append(direction + "," + training_row)
    validation_rows.append(direction + "," + validation_row)

with open("results/training_results.csv", "w") as f:
    f.write("\n".join(training_rows))

with open("results/validation_results.csv", "w") as f:
    f.write("\n".join(validation_rows))

