 # -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:45:57 2018

@author: Alvaro
"""

from flask import Flask, render_template, request, redirect, make_response, send_file, flash
import os
import pandas as pd
import numpy as np
import models as algorithms
import plotfunctions as plotfun
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask_bootstrap import Bootstrap

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from validate_email import validate_email
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

bootstrap = Bootstrap(app)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'datasets')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Config for SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150))
    last_name = db.Column(db.String(150))
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(200))

# Create the DB once
with app.app_context():
    db.create_all()

# Load user by ID (used by Flask-Login)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



def datasetList():
    datasets = [x.split('.')[0] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    folders = [f for f in ['datasets', 'preprocessed'] for x in os.listdir(f)]
    return datasets, extensions, folders

#Load columns of the dataset
def loadColumns(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'), nrows=0)
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'), nrows=0)
        return df.columns

#Load Dataset    
def loadDataset(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'))
        return df


@app.route('/', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect('/index')

    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password1 = request.form['password1']
        password2 = request.form['password2']

        if len(password1) < 6:
            flash('⚠️ Password should be at least 6 characters', 'danger')
            return redirect('/')

        if password1 != password2:
            flash('⚠️ Password Mismatch!', 'danger')
            return redirect('/')

        if not validate_email(email):
            flash('⚠️ Invalid Email Address', 'danger')
            return redirect('/')

        if not username:
            flash('⚠️ Username is required!', 'danger')
            return redirect('/')

        if User.query.filter_by(username=username).first():
            flash('⚠️ Username is already taken!', 'danger')
            return redirect('/')

        if User.query.filter_by(email=email).first():
            flash('⚠️ Email is already taken!', 'danger')
            return redirect('/')

        hashed_password = generate_password_hash(password1)
        new_user = User(first_name=first_name, last_name=last_name, email=email,
                        username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('✅ Registration successful! You can now login.', 'success')
        return redirect('/login')

    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect('/index')

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if not user or not check_password_hash(user.password, password):
            flash('⚠️ Invalid credentials, try again', 'danger')
            return render_template('login.html')

        login_user(user)
        flash('✅ Login successful!', 'success')
        return redirect('/index')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('✅ Successfully Logged Out!', 'success')
    return redirect('/login')


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    datasets, _, folders = datasetList()
    originalds = []
    featuresds = []

    # Split datasets into originals and preprocessed
    for i in range(len(datasets)):
        if folders[i] == 'datasets':
            originalds.append(datasets[i])
        else:
            featuresds.append(datasets[i])

    # Handle file upload
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('⚠️ No file part in the request', 'warning')
            return redirect(request.url)

        f = request.files['file']

        if f.filename == '':
            flash('⚠️ No file selected', 'warning')
            return redirect(request.url)

        # Save file inside 'datasets' folder
        filepath = os.path.join('datasets', f.filename)
        f.save(filepath)
        flash(f'✅ File "{f.filename}" uploaded successfully!', 'success')
        return redirect('/index')


    return render_template(
        'index.html',
        originalds=originalds,
        featuresds=featuresds
    )


@app.route('/datasets/')
def datasets():
    return redirect('/')

@app.route('/datasets/<dataset>')
def dataset(description = None, head = None, dataset = None):
    df = loadDataset(dataset)
    try:
        description = df.describe().round(2)
        head = df.head(5)
    except: pass
    return render_template('dataset.html',
                           description = description.to_html(classes='table table-striped table-hover'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)

@app.route('/datasets/<dataset>/models')
def models(dataset = dataset):
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('models.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels,
                           columns = columns)

@app.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
def model_process(dataset = dataset):
    algscore = request.form.get('model')
    res = request.form.get('response')
    kfold = request.form.get('kfold')
    alg, score = algscore.split('.')
    scaling = request.form.get('scaling')
    variables = request.form.getlist('variables')
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    df = loadDataset(dataset)
    y = df[str(res)]

    if variables != [] and '' not in variables: df = df[list(set(variables + [res]))]
    X = df.drop(str(res), axis=1)
    try: X = pd.get_dummies(X)
    except: pass
    
    predictors = X.columns
    if len(predictors)>10: pred = str(len(predictors))
    else: pred = ', '.join(predictors)    

    if score == 'Classification':
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
        scoring = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']
        if scaling == 'Yes':
            clf = algorithms.classificationModels()[alg]
            mod = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        else: 
            mod = algorithms.classificationModels()[alg]
        fig = plotfun.plot_ROC(X.values, y, mod, int(kfold))

    elif score == 'Regression':
        from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
        scoring = ['explained_variance', 'r2', 'mean_squared_error']
        if scaling == 'Yes':
            pr = algorithms.regressionModels()[alg]  
            mod = Pipeline([('scaler', StandardScaler()), ('clf', pr)])
        else: mod = algorithms.regressionModels()[alg]
        fig = plotfun.plot_predVSreal(X, y, mod, int(kfold))
    
    scores = cross_validate(mod, X, y, cv=int(kfold), scoring=scoring)
    for s in scores:
        scores[s] = str(round(np.mean(scores[s]),3))
    return render_template('scores.html', scores = scores, dataset = dataset, alg=alg,
                           res = res, kfold = kfold, score = score,
                           predictors = pred, response = str(fig, 'utf-8'))
    
@app.route('/datasets/<dataset>/preprocessing')
def preprocessing(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('preprocessing.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/preprocessed_dataset/', methods=['POST'])
def preprocessed_dataset(dataset):
    numFeatures = request.form.get('nfeatures')
    manualFeatures = request.form.getlist('manualfeatures')
    datasetName = request.form.get('newdataset')
    response = request.form.get('response')
    dropsame = request.form.get('dropsame')
    dropna = request.form.get('dropna')
    
    df = loadDataset(dataset)

    if dropna == 'all':
        df = df.dropna(axis=1, how='all')
    elif dropna == 'any':
        df.dropna(axis=1, how='any')
        
    filename = dataset + '_'
    try:
        nf = int(numFeatures)
        from sklearn.feature_selection import SelectKBest, chi2
        X = df.drop(str(response), axis=1)
        y = df[str(response)]
        kbest = SelectKBest(chi2, k=nf).fit(X,y)
        mask = kbest.get_support()
        # List of K best features
        best_features = []
        for bool, feature in zip(mask, list(df.columns)):
            if bool: best_features.append(feature)
        #Reduced Dataset
        df = pd.DataFrame(kbest.transform(X),columns=best_features)
        df.insert(0, str(response), y)
        
        filename += numFeatures + '_' + 'NA' + dropna + '_Same' + dropsame + '.csv'
    
    except:
        df = df[manualFeatures]
        filename += str(datasetName) + '_' + str(response) + '.csv'
    
    if dropsame == 'Yes':
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)
    df.to_csv(os.path.join('preprocessed', filename), index=False)
    return redirect('/datasets/' + filename.split('.')[0])


@app.route('/datasets/<dataset>/graphs')
def graphs(dataset = dataset):
    columns = loadColumns(dataset)
    return render_template('graphs.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/graphprocess/', methods=['POST'])
def graph_process(dataset=dataset):
    histogram = request.form.getlist('histogram')
    boxplotcat = request.form.get('boxplotcat')
    boxplotnum = request.form.get('boxplotnum')
    corr = request.form.getlist('corr')
    corrcat = request.form.get('corrcat')

    ds = loadDataset(dataset)
    import plotfunctions as plotfun
    figs = {}

    # 1. Histogram
    if histogram and any(h.strip() != "" for h in histogram):
        figs['Histograms'] = str(plotfun.plot_histsmooth(ds, histogram), 'utf-8')

    # 2. Correlation Plot (check for empty safely)
    if corr and any(c.strip() != "" for c in corr):
        try:
            figs['Correlations'] = str(plotfun.plot_correlations(ds, corr, corrcat), 'utf-8')
        except Exception as e:
            print(f"[Correlation Error] {e}")
            figs['Correlations'] = "<p style='color:red;'>Invalid correlation selection: Please select at least one numeric variable.</p>"

    # 3. Box Plot
    if boxplotcat != '' and boxplotnum != '':
        figs['Box Plot'] = str(plotfun.plot_boxplot(ds, boxplotcat, boxplotnum), 'utf-8')

    # 4. No plots selected
    if figs == {}:
        return redirect(f'/datasets/{dataset}/graphs')

    return render_template('drawgraphs.html', figs=figs, dataset=dataset)

@app.route('/datasets/<dataset>/predict')
def predict(dataset = dataset):
    columns = loadColumns(dataset)
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('predict.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels,
                           columns = columns)

@app.route('/datasets/<dataset>/prediction/', methods=['POST'])
def predict_process(dataset = dataset):
    algscore = request.form.get('model')
    res = request.form.get('response')
    alg, score = algscore.split('.')
    scaling = request.form.get('scaling')
    df = loadDataset(dataset)
    columns = df.columns
    values = {}
    counter = 0
    for col in columns:
        values[col] = request.form.get(col)
        if values[col] != '' and col != res: counter +=1
    
    if counter == 0: return redirect('/datasets/' + dataset + '/predict')
    
    predictors = {}
    for v in values:
        if values[v] != '':
            try: predictors[v] = [float(values[v])]
            except: predictors[v] = [values[v]]

    from sklearn.preprocessing import StandardScaler
    X = df[list(predictors.keys())]
    Xpred = predictors
    #return str(Xpred)
    Xpred = pd.DataFrame(data=Xpred)
    X = pd.concat([X,Xpred])
    X = pd.get_dummies(X)
    Xpred = X.iloc[[-1]]
    X = X[:-1]
    if scaling == 'Yes':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
        Xpred = pd.DataFrame(scaler.transform(Xpred), columns = X.columns)
    try:
        X = X.drop(str(res), axis=1)
        Xpred = Xpred.drop(str(res), axis=1)
    except: pass
    #Xpred.reset_index(drop=True, inplace=True)
    #X.reset_index(drop=True, inplace=True)
    y = df[str(res)]
    if score == 'Classification':
            mod = algorithms.classificationModels()[alg]
    elif score == 'Regression':
        mod = algorithms.regressionModels()[alg]
    model = mod.fit(X, y)
    #return pd.DataFrame(Xpred).to_html()
    predictions = {}
    predictions['Prediction'] = model.predict(Xpred)[0]
    predictors.pop(res, None)
    for p in predictors:
        if str(predictors[p][0]).isdigit() is True: predictors[p] = int(predictors[p][0])
        else:
            try: predictors[p] = round(predictors[p][0],2)
            except: predictors[p] = predictors[p][0]
    for p in predictions:
        if str(predictions[p]).isdigit() is True: predictions[p] = int(predictions[p])
        else:
            try: predictions[p] = round(predictions[p],2)
            except: continue
    if len(predictors) > 15: predictors = {'Number of predictors': len(predictors)}
    #return str(predictors) + res + str(predictions) + alg + score
    if score == 'Classification':
        classes = model.classes_
        pred_proba = model.predict_proba(Xpred)
        for i in range(len(classes)):
            predictions['Prob. ' + str(classes[i])] = round(pred_proba[0][i],3)    
    return render_template('prediction.html', predictions = predictions, response = res,
                           predictors = predictors, algorithm = alg, score = score,
                           dataset = dataset)

@app.errorhandler(500)
def internal_error(e):
    return render_template('error500.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error404.html')

if __name__ == '__main__':
    app.run(debug=False)
