# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:29:55 2018

@author: Alvaro
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

paths = {
    'output': 'static/output'
}
os.makedirs(paths['output'], exist_ok=True) 
#Plot ROC curves
def plot_ROC(X, y, classifier, cv):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp
    cv = StratifiedKFold(n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    #figure = plt.figure()
    plt.gcf().clear()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png
    
def plot_predVSreal(X, y, classifier, cv):
    from sklearn.model_selection import cross_val_predict
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(classifier, X, y, cv=cv)
    plt.gcf().clear()
    plt.scatter(y, predicted, edgecolors=(0, 0, 0))
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_histsmooth(ds, columns):
    sns.set()
    plt.gcf().clear()
    for col in columns:
        sns.distplot(ds[col], label = col)
    from io import BytesIO
    plt.xlabel('')
    plt.legend()
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64

def plot_correlations(ds, corr, corrcat):
    sns.set()
    plt.gcf().clear()

    # Validate 'corr' input
    if not corr or all(col.strip() == "" for col in corr):
        raise ValueError("No variables selected for correlation plot.")

    # Filter only numeric columns from 'corr'
    valid_corr = [col for col in corr if col in ds.columns and ds[col].dtype.kind in "iufc"]
    if not valid_corr:
        raise ValueError("No valid numeric columns found in selected correlation variables.")

    # Try to create pairplot
    try:
        if corrcat and corrcat in ds.columns:
            sns.pairplot(ds[valid_corr + [corrcat]], hue=corrcat)
        else:
            sns.pairplot(ds[valid_corr])
    except Exception as e:
        raise ValueError(f"Failed to plot correlations: {e}")

    # Save plot
    plot_path = os.path.join(paths['output'], 'correlation.png')
    plt.savefig(plot_path)
    plt.close()

    # Encode as base64 to embed in HTML
    with open(plot_path, "rb") as image_file:
        return base64.b64encode(image_file.read())



def plot_boxplot(ds, cat, num):
    sns.set()
    plt.gcf().clear()

    # Validate columns
    if cat not in ds.columns or num not in ds.columns:
        raise ValueError("Selected variables are not in dataset.")

    # Ensure categorical and numerical types
    if not (ds[cat].dtype == "object" or str(ds[cat].dtype).startswith("category")):
        ds[cat] = ds[cat].astype(str)
    if ds[num].dtype.kind not in "iufc":
        raise ValueError("Selected numerical variable must be numeric.")

    # Plot using updated API
    g = sns.catplot(x=cat, y=num, data=ds, kind="box", height=6, aspect=1.5)
    g.set_xticklabels(rotation=45)

    # Save to file
    plot_path = os.path.join(paths['output'], 'boxplot.png')
    plt.savefig(plot_path)
    plt.close()

    # Return as base64
    with open(plot_path, "rb") as image:
        return base64.b64encode(image.read())

    
    


        