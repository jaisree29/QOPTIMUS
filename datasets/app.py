# app.py
from flask import Flask, render_template_string, request, redirect, flash
import os
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --- Basic Flask setup ---
app = Flask(__name__)
app.secret_key = "change_this_secret"
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'datasets')
PREPROC_FOLDER = os.path.join(os.getcwd(), 'preprocessed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROC_FOLDER, exist_ok=True)

ALLOWED_EXT = {'csv'}

# --- Helper functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def dataset_list():
    # return dataset filenames without extension
    datasets = []
    folders = []
    for folder in ['datasets','preprocessed']:
        path = os.path.join(os.getcwd(), folder)
        for f in os.listdir(path):
            if f.lower().endswith('.csv') or f.lower().endswith('.txt'):
                datasets.append(os.path.splitext(f)[0])
                folders.append(folder)
    return datasets, folders

def load_dataset(name):
    # look in datasets then preprocessed
    for folder in ['datasets','preprocessed']:
        for ext in ['.csv','.txt']:
            p = os.path.join(os.getcwd(), folder, name + ext)
            if os.path.exists(p):
                if ext == '.csv':
                    return pd.read_csv(p)
                else:
                    return pd.read_table(p)
    return None

def load_columns(name):
    df = load_dataset(name)
    if df is None:
        return []
    return list(df.columns)

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

def drop_identifier_columns(df):
    # drop obvious id columns (name 'id' or unique values for every row)
    to_drop = []
    for col in df.columns:
        if col.lower()=='id':
            to_drop.append(col)
        elif df[col].nunique() == len(df):
            to_drop.append(col)
    return df.drop(columns=to_drop, errors='ignore'), to_drop

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return "data:image/png;base64," + data

# --- Simple models collection ---
def classification_models():
    return {'logreg': LogisticRegression(max_iter=200), 'rf': RandomForestClassifier(n_estimators=50)}

def regression_models():
    return {'linreg': LinearRegression(), 'rfreg': RandomForestRegressor(n_estimators=50)}

# --- Templates (render_template_string so single file) ---
INDEX_HTML = """
<!doctype html>
<title>CoolScience - Home</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>CoolScience - Upload dataset (CSV)</h2>
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for cat,msg in messages %}
      <div class="alert alert-{{cat}}">{{msg}}</div>
    {% endfor %}
  {% endwith %}
  <form method="post" enctype="multipart/form-data">
    <div class="form-group">
      <input type="file" name="file" required>
    </div>
    <button class="btn btn-primary">Upload</button>
  </form>
  <hr>
  <h4>Available datasets</h4>
  <ul>
    {% for d in datasets %}
      <li><a href="/dataset/{{d}}">{{d}}</a></li>
    {% else %}
      <li>No datasets yet. Upload one above.</li>
    {% endfor %}
  </ul>
</div>
"""

DATASET_HTML = """
<!doctype html>
<title>Dataset: {{dataset}}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>Dataset: {{dataset}}</h2>
  {% if head_html %}
    <h4>First rows</h4>
    {{ head_html | safe }}
    <h4>Summary</h4>
    {{ desc_html | safe }}
  {% else %}
    <p class="text-danger">Could not load dataset.</p>
  {% endif %}
  <p>
    <a class="btn btn-default" href="/dataset/{{dataset}}/preprocess">Preprocessing</a>
    <a class="btn btn-default" href="/dataset/{{dataset}}/graphs">Graphs</a>
    <a class="btn btn-default" href="/dataset/{{dataset}}/models">Models</a>
    <a class="btn btn-default" href="/dataset/{{dataset}}/predict">Predict</a>
    <a class="btn btn-primary" href="/">Back</a>
  </p>
</div>
"""

PREPROC_HTML = """
<!doctype html>
<title>Preprocessing - {{dataset}}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>Preprocessing: {{dataset}}</h2>
  <form method="post">
    <div class="form-group">
      <label>Response (target) column:</label>
      <select name="response" class="form-control">
        <option value="">-- select target --</option>
        {% for c in columns %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Manual select variables to KEEP (hold Ctrl to multi-select). Leave empty to keep all numeric + dummies.</label>
      <select name="manual" multiple class="form-control" size="6">
        {% for c in columns %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="checkbox">
      <label><input type="checkbox" name="drop_unique" value="1"> Drop columns with a single unique value</label>
    </div>
    <div class="form-group">
      <label>Drop rows with nulls:</label>
      <select name="dropna" class="form-control">
        <option value="">No</option>
        <option value="any">If any column null</option>
        <option value="all">If all columns null</option>
      </select>
    </div>
    <button class="btn btn-success">Create preprocessed dataset</button>
    <a class="btn btn-default" href="/dataset/{{dataset}}">Back</a>
  </form>
</div>
"""

GRAPHS_HTML = """
<!doctype html>
<title>Graphs - {{dataset}}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>Graphs: {{dataset}}</h2>
  <form method="post">
    <div class="form-group">
      <label>Histograms (select numeric columns):</label>
      <select name="hist" multiple class="form-control" size="6">
        {% for c in numeric %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Boxplot - Categorical (optional):</label>
      <select name="boxcat" class="form-control">
        <option value="">-- none --</option>
        {% for c in categorical %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
      <label>Boxplot - Numeric:</label>
      <select name="boxnum" class="form-control">
        <option value="">-- none --</option>
        {% for c in numeric %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Correlation (select numeric columns):</label>
      <select name="corr" multiple class="form-control" size="6">
        {% for c in numeric %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <button class="btn btn-primary">Show graphs</button>
    <a class="btn btn-default" href="/dataset/{{dataset}}">Back</a>
  </form>

  {% if imgs %}
    <hr>
    <h3>Plots</h3>
    {% for title,src in imgs.items() %}
      <h4>{{title}}</h4>
      <img src="{{src}}" style="max-width:90%;border:1px solid #ddd;padding:4px;margin-bottom:20px;">
    {% endfor %}
  {% endif %}
</div>
"""

MODELS_HTML = """
<!doctype html>
<title>Models - {{dataset}}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>Models: {{dataset}}</h2>
  <form method="post">
    <div class="form-group">
      <label>Select response (target):</label>
      <select name="response" class="form-control" required>
        <option value="">-- choose --</option>
        {% for c in columns %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Variables to use (leave empty to use all except id/target):</label>
      <select name="vars" multiple class="form-control" size="8">
        {% for c in columns %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Model type:</label>
      <select name="model" class="form-control">
        <option value="classification">Classification</option>
        <option value="regression">Regression</option>
      </select>
    </div>
    <div class="form-group">
      <label>Algorithm:</label>
      <select name="algo" class="form-control">
        <option value="logreg">Logistic Regression (classification)</option>
        <option value="rf">RandomForestClassifier</option>
        <option value="linreg">LinearRegression (regression)</option>
        <option value="rfreg">RandomForestRegressor</option>
      </select>
    </div>
    <div class="form-group">
      <label>k-fold CV (integer):</label>
      <input name="k" type="number" class="form-control" value="5">
    </div>
    <button class="btn btn-success">Run model (cross-validate)</button>
    <a class="btn btn-default" href="/dataset/{{dataset}}">Back</a>
  </form>

  {% if scores %}
    <hr>
    <h3>CV Results</h3>
    <pre>{{scores}}</pre>
  {% endif %}
</div>
"""

PREDICT_HTML = """
<!doctype html>
<title>Predict - {{dataset}}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2>Predict: {{dataset}}</h2>
  <form method="post">
    <div class="form-group">
      <label>Response (target):</label>
      <select name="response" class="form-control" required>
        {% for c in columns %}
          <option value="{{c}}">{{c}}</option>
        {% endfor %}
      </select>
    </div>
    <div class="form-group">
      <label>Algorithm:</label>
      <select name="algo" class="form-control">
        <option value="logreg">logreg</option>
        <option value="rf">rf</option>
        <option value="linreg">linreg</option>
        <option value="rfreg">rfreg</option>
      </select>
    </div>
    <h4>Enter predictor values (leave blank to skip)</h4>
    {% for c in columns %}
      {% if c != 'id' %}
      <div class="form-group">
        <label>{{c}}</label>
        <input name="{{c}}" class="form-control" placeholder="value for {{c}}">
      </div>
      {% endif %}
    {% endfor %}
    <button class="btn btn-primary">Run prediction</button>
    <a class="btn btn-default" href="/dataset/{{dataset}}">Back</a>
  </form>

  {% if result %}
    <hr>
    <h3>Prediction result</h3>
    <pre>{{ result }}</pre>
  {% endif %}
</div>
"""

ERROR_HTML = """
<!doctype html>
<title>Error</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css">
<div class="container">
  <h2 class="text-danger">An error occurred</h2>
  <pre>{{err}}</pre>
  <a class="btn btn-default" href="/">Back</a>
</div>
"""

# --- Routes ---
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or f.filename=='':
            flash('No file selected','warning')
            return redirect('/')
        if not allowed_file(f.filename):
            flash('Only CSV files allowed','danger')
            return redirect('/')
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        flash(f'Uploaded {f.filename}','success')
        return redirect('/')
    datasets, _ = dataset_list()
    return render_template_string(INDEX_HTML, datasets=datasets)

@app.route('/dataset/<dataset>')
def dataset_page(dataset):
    try:
        df = load_dataset(dataset)
        if df is None:
            return render_template_string(DATASET_HTML, dataset=dataset, head_html=None, desc_html=None)
        head_html = df.head(5).to_html(classes='table table-striped', index=False)
        desc = df.describe(include='all').round(3)
        desc_html = desc.to_html(classes='table table-bordered')
        return render_template_string(DATASET_HTML, dataset=dataset, head_html=head_html, desc_html=desc_html)
    except Exception as e:
        return render_template_string(ERROR_HTML, err=str(e))

@app.route('/dataset/<dataset>/preprocess', methods=['GET','POST'])
def preprocess(dataset):
    try:
        columns = load_columns(dataset)
        if request.method == 'POST':
            response = request.form.get('response') or ''
            manual = request.form.getlist('manual')
            drop_unique = request.form.get('drop_unique')
            dropna = request.form.get('dropna')

            df = load_dataset(dataset)
            if df is None:
                flash('Dataset not found','danger')
                return redirect(f'/dataset/{dataset}')

            # Drop id-like columns
            df, dropped = drop_identifier_columns(df)

            # Drop rows with nulls if requested
            if dropna == 'any':
                df = df.dropna(axis=0, how='any')
            elif dropna == 'all':
                df = df.dropna(axis=0, how='all')

            # choose columns to keep
            if manual:
                # ensure response included
                if response and response not in manual:
                    manual.append(response)
                df = df[manual]

            if drop_unique:
                nunique = df.nunique()
                cols_to_drop = list(nunique[nunique <= 1].index)
                df = df.drop(columns=cols_to_drop, errors='ignore')

            # Save preprocessed with a safe name
            safe_name = f"{dataset}_preproc.csv"
            df.to_csv(os.path.join(PREPROC_FOLDER, safe_name), index=False)
            flash(f'Preprocessed saved as {safe_name}','success')
            return redirect(f'/dataset/{os.path.splitext(safe_name)[0]}')
        return render_template_string(PREPROC_HTML, dataset=dataset, columns=columns)
    except Exception as e:
        return render_template_string(ERROR_HTML, err=str(e))

@app.route('/dataset/<dataset>/graphs', methods=['GET','POST'])
def graphs(dataset):
    try:
        df = load_dataset(dataset)
        if df is None:
            flash('Dataset not found','danger')
            return redirect('/')
        # prepare numeric and categorical lists
        numeric = [c for c in df.columns if is_numeric_series(df[c])]
        categorical = [c for c in df.columns if not is_numeric_series(df[c])]
        imgs = {}
        if request.method == 'POST':
            hist = request.form.getlist('hist')
            boxcat = request.form.get('boxcat') or ''
            boxnum = request.form.get('boxnum') or ''
            corr = request.form.getlist('corr')

            # 1. histograms
            if hist:
                # ensure only numeric selected
                hist_numeric = [h for h in hist if h in numeric]
                if not hist_numeric:
                    imgs['Histograms'] = "<p style='color:red;'>No numeric column selected for histograms.</p>"
                else:
                    fig, ax = plt.subplots(figsize=(8,3))
                    df[hist_numeric].hist(ax=ax, bins=20)
                    # when multiple axes return object - handle generically by making a new fig per column
                    if hasattr(ax, '__iter__'):
                        plt.close(fig)
                        for col in hist_numeric:
                            fig2 = plt.figure(figsize=(4,3))
                            df[col].hist(bins=25)
                            plt.title(col)
                            imgs[f'Histogram - {col}'] = fig_to_base64(plt.gcf())
                    else:
                        imgs['Histograms'] = fig_to_base64(fig)

            # 2. correlations
            if corr:
                corr_numeric = [c for c in corr if c in numeric]
                if len(corr_numeric) < 2:
                    imgs['Correlation'] = "<p style='color:red;'>Select at least two numeric columns for correlation.</p>"
                else:
                    fig = plt.figure(figsize=(6,5))
                    sub = df[corr_numeric].corr()
                    plt.matshow(sub, fignum=fig.number)
                    plt.xticks(range(len(corr_numeric)), corr_numeric, rotation=90)
                    plt.yticks(range(len(corr_numeric)), corr_numeric)
                    plt.colorbar()
                    imgs['Correlation matrix'] = fig_to_base64(fig)

            # 3. boxplot
            if boxnum:
                if boxnum not in numeric:
                    imgs['Boxplot'] = "<p style='color:red;'>Choose a numeric column for the boxplot.</p>"
                else:
                    fig = plt.figure(figsize=(6,4))
                    if boxcat and boxcat in df.columns:
                        # try to plot grouped boxplot
                        df.boxplot(column=boxnum, by=boxcat)
                        plt.title(f'{boxnum} by {boxcat}')
                        plt.suptitle('')
                    else:
                        df.boxplot(column=boxnum)
                    imgs['Boxplot'] = fig_to_base64(fig)

        return render_template_string(GRAPHS_HTML, dataset=dataset, numeric=numeric, categorical=categorical, imgs=imgs)
    except Exception as e:
        return render_template_string(ERROR_HTML, err=str(e))

@app.route('/dataset/<dataset>/models', methods=['GET','POST'])
def models_route(dataset):
    try:
        df = load_dataset(dataset)
        if df is None:
            flash('Dataset not found','danger')
            return redirect('/')
        columns = list(df.columns)
        scores = None
        if request.method == 'POST':
            response = request.form.get('response')
            vars_sel = request.form.getlist('vars')
            model_type = request.form.get('model')
            algo = request.form.get('algo')
            k = int(request.form.get('k') or 5)

            if response == '' or response not in df.columns:
                flash('Choose a valid response/target column','danger')
                return redirect(f'/dataset/{dataset}/models')

            # build X,y
            if vars_sel:
                use_cols = [c for c in vars_sel if c in df.columns and c != response]
            else:
                # default: all columns except id and response
                tmp, dropped = drop_identifier_columns(df.copy())
                use_cols = [c for c in tmp.columns if c != response]

            X = df[use_cols].copy()
            # convert categoricals with dummies, numeric left as-is
            X = pd.get_dummies(X, drop_first=True)
            y = df[response].copy()
            # if y is non-numeric, try to map to ints
            if not is_numeric_series(y):
                try:
                    y = pd.Series(pd.Categorical(y)).codes
                except:
                    pass

            # choose model
            if model_type == 'classification':
                algs = classification_models()
            else:
                algs = regression_models()

            if algo not in algs:
                flash('Chosen algorithm incompatible or not available','danger')
                return redirect(f'/dataset/{dataset}/models')

            mod = algs[algo]
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', mod)]) if X.shape[1]>0 else mod

            # run cross-validate safely
            scoring = None
            if model_type == 'classification':
                scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
            else:
                scoring = ['r2','neg_mean_squared_error']
            try:
                scores = cross_validate(pipeline, X.fillna(0), y, cv=k, scoring=scoring)
                # summarize
                scores_summary = {k: float(np.mean(v)) for k,v in scores.items()}
                scores = scores_summary
            except Exception as e:
                scores = {'error': str(e)}

        return render_template_string(MODELS_HTML, dataset=dataset, columns=columns, scores=scores)
    except Exception as e:
        return render_template_string(ERROR_HTML, err=str(e))

@app.route('/dataset/<dataset>/predict', methods=['GET','POST'])
def predict_route(dataset):
    try:
        df = load_dataset(dataset)
        if df is None:
            flash('Dataset not found','danger')
            return redirect('/')
        columns = list(df.columns)
        result = None
        if request.method == 'POST':
            response = request.form.get('response')
            algo = request.form.get('algo')

            # collect predictor values
            predictors = {}
            for c in columns:
                if c == 'id': continue
                val = request.form.get(c)
                if val is None or val.strip()=='':
                    continue
                # try convert to float
                try:
                    v = float(val)
                    predictors[c] = v
                except:
                    predictors[c] = val

            # build X from dataset columns used for predictors
            if not predictors:
                flash('Enter at least one predictor value','warning')
                return redirect(f'/dataset/{dataset}/predict')

            X_full = df[list(predictors.keys())].copy()
            X_full = pd.get_dummies(X_full, drop_first=True)
            # create DataFrame for new sample
            new_df = pd.DataFrame([predictors])
            new_df = pd.get_dummies(new_df, drop_first=True)

            # align columns
            X_full, new_df = X_full.align(new_df, join='outer', axis=1, fill_value=0)

            # get target y if exists
            if response and response in df.columns:
                y = df[response]
                if not is_numeric_series(y):
                    try:
                        y = pd.Series(pd.Categorical(y)).codes
                    except:
                        pass
            else:
                # if no response provided try to infer nothing
                y = None

            # choose alg
            all_algs = {**classification_models(), **regression_models()}
            if algo not in all_algs:
                flash('Choose a valid algorithm','danger')
                return redirect(f'/dataset/{dataset}/predict')

            model = all_algs[algo]
            # fit if we have y and columns match
            try:
                if y is not None:
                    model.fit(X_full.fillna(0), y)
                    pred = model.predict(new_df.fillna(0))[0]
                else:
                    # unsupervised/predict with training of dummy
                    model.fit(X_full.fillna(0), np.zeros(len(X_full)))
                    pred = model.predict(new_df.fillna(0))[0]
                result = {'prediction': str(pred)}
                # If classifier, try probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(new_df.fillna(0))[0]
                    result['proba'] = list(map(float, proba))
            except Exception as e:
                result = {'error': str(e)}

        return render_template_string(PREDICT_HTML, dataset=dataset, columns=columns, result=result)
    except Exception as e:
        return render_template_string(ERROR_HTML, err=str(e))

# Custom error handlers
@app.errorhandler(500)
def internal_server_error(e):
    return render_template_string(ERROR_HTML, err=str(e)), 500

@app.errorhandler(404)
def not_found(e):
    return render_template_string(ERROR_HTML, err='Page not found'), 404

if __name__ == '__main__':
    # set debug=False in production; debug=True will show tracebacks in browser (useful to debug)
    app.run(debug=True, port=5000)
