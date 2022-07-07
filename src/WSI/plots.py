import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
from varname import nameof
import seaborn as sn

"""
Code used to get plots for confusion matrices and learning curve

"""
def learning_curve_train(NAME):
  fig, axs = plt.subplots(2, figsize = (7, 6))

  acc_df = pd.read_csv(f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\results\\WSI\\log\\model_{NAME}.log")
  acc_df.columns = ["MODEL_NAME", "TIME", "ACC", "LOSS", "CONF_M", "AUC",
                     "VAL_ACC", "VAL_LOSS", "VAL_CONF_M", "VAL_AUC"]

  fig, axs = plt.subplots(2, figsize=(5,7))

  axs[0].legend("MODEL_NAME", loc=2)

  acc_df.plot(y="ACC", ax=axs[0])
  acc_df.plot(y="VAL_ACC", ax=axs[0])

  acc_df.plot(y="LOSS", ax=axs[1])
  acc_df.plot(y="VAL_LOSS",ax=axs[1])

  fig.show()
  fig.savefig(f"C:\\Users\\Alejandro\\Desktop\\heterogeneous-data\\results\\WSI\\lc\\l_curve_{NAME}.pdf")


def learning_curve(direc, MODEL_NAMES):
    fig, axs = plt.subplots(2, figsize = (7,8))

    acc_, val_acc_, loss_, val_loss_ = [],[],[],[]

    for MODEL_NAME in MODEL_NAMES:

        acc_df = pd.read_csv(f"{direc}/model_{MODEL_NAME}.log",
                            names=["MODEL_NAME", "TIME", "ACC", "LOSS", "CONF_M", "PRC", "REC",
                     "VAL_ACC", "VAL_LOSS", "VAL_CONF_M", "VAL_PRC", "VAL_REC"])
        
        acc_df = acc_df.loc[acc_df['MODEL_NAME'] == MODEL_NAME]

        acc = acc_df["ACC"].to_numpy()
        loss = acc_df["LOSS"].to_numpy()

        val_acc = acc_df["VAL_ACC"].to_numpy()
        val_loss = acc_df["VAL_LOSS"].to_numpy()
    
        acc_.append(acc)
        loss_.append(loss)

        val_acc_.append(val_acc)
        val_loss_.append(val_loss)

    acc = np.mean(acc_, axis=0)
    loss = np.mean(loss_, axis=0)
    val_acc = np.mean(val_acc_, axis=0)
    val_loss = np.mean(val_loss_, axis=0)

    acc_err = np.std(acc_, axis=0)
    loss_err = np.std(loss_, axis=0)
    val_acc_err = np.std(val_acc_, axis=0)
    val_loss_err = np.std(val_loss_, axis=0)

    t = range(len(acc))
    
    axs[0].legend("MODEL_NAME", loc=2)

    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")

    axs[0].plot(t,acc, 'o' , ls ="-", label= "Training")
    axs[0].plot(t,val_acc, 'o', ls ="-", label= "Validation")


    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Binary cross entropy")

    axs[1].plot(t,loss, 'o', ls ="-", label= "Training")
    axs[1].plot(t,val_loss, 'o', ls ="-", label= "Validation")


    for idx, i in enumerate([loss, val_loss, acc , val_acc]):
        x = t
        y = i
        x_list = np.linspace(0,20,100)
        poly = np.polyfit(x, y,5)
        poly_y = np.poly1d(poly)(x_list)

        if idx<2:
            if nameof(i)==nameof(acc):
                err = acc_err
            else:
                err = val_acc_err
            #axs[1].plot(x_list, poly_y)
            axs[1].fill_between(t, y + err,
                     y - err, alpha=0.1,)
        else:
            if nameof(i)==nameof(loss):
                err = loss_err
            else:
                err = val_loss_err
            #axs[0].plot(x_list, poly_y)
            axs[0].fill_between(t, y + err,
                     y - err, alpha=0.1,)

    axs[0].legend(loc = "lower right")
    axs[1].legend()
    fig.show()
    fig.savefig(f"C:/Users/Alejandro/Desktop/heterogeneous-data/results/WSI/lc/l_curve_{MODEL_NAME[0]}.pdf")


def read_conf_matrix(SPLIT_NAME, EPOCH):
    acc_df = pd.read_csv(f"C:/Users/Alejandro/Desktop/heterogeneous-data/results/WSI/log/model_{SPLIT_NAME}.log",
                        names=["MODEL_NAME", "TIME", "ACC", "LOSS", "CONF_M", "PRC", "REC",
                        "VAL_ACC", "VAL_LOSS", "VAL_CONF_M", "VAL_PRC", "VAL_REC"])

    conf_list = [int(i) for i in acc_df["CONF_M"][EPOCH].split("+")]
    conf = np.reshape(conf_list, (2,2))

    return conf


def plot_conf(SPLIT_NAME, conf):

    fig, ax = plt.subplots()
        
    lab = ["Negative", "Positive"]

    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf.flatten()/np.sum(conf)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sn.heatmap(conf, annot=labels, fmt='', cmap='Blues', ax= ax)

    ax.set_xticklabels(lab)
    ax.set_yticklabels(lab)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    fig.savefig(f"C:/Users/Alejandro/Desktop/heterogeneous-data/results/WSI/lc/conf_matrix_{SPLIT_NAME}.pdf")

def plot_roc(SPLIT_NAME, roc):

    from sklearn.metrics import auc

    auc = auc(np.linspace(0,1,len(roc)), roc)

    fig, ax = plt.subplots()

    ax.set_xlabel("FPR")
    ax.set_ylabel("TNR")

    ax.set_title("ROC")

    ax.plot(np.linspace(0,1,len(roc)), roc, label="AUC="+str(round(auc,3)))
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), 'k--')

    ax.legend()

    fig.savefig(f"C:/Users/Alejandro/Desktop/heterogeneous-data/results/WSI/lc/roc_{SPLIT_NAME}.pdf")

