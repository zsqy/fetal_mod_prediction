import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import sys

from scipy import stats


#TODO: centile
#TODO: balanced accuracy is low
#TODO: hyperparam tuning
#TODO: feature importance
#TODO: feature importance ml

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler()
handler = logging.FileHandler('log.txt')
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

# data file
antenatal_data = 'data/FGR_STUDY_20_21.csv'
postnatal_data = 'data/IUGR_Studies.csv'

# to get rid of tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


####################
# preprocess
####################

def process_antenatal_data():
    logger.info('process antenatal data')
    df = pd.read_csv(antenatal_data, index_col=4)
    logger.debug('\tdimension, original df: {}'.format(df.shape))
    
    # combine GA week and day
    df['ga'] = df['GA (WK)'] + df['GA (DAYS)'] / 7
    
    # drop PID, case number, exam, exam date as they are unrelated
    # drop GA week and days as already have ga new column
    # drop ethnic group, tav, uterine, RI, notch because very less data
    df.drop(columns=[
        'PID', 
        'Case number', 
        'Exam',
        'Examination date',
        'GA (WK)',
        'GA (DAYS)',
        'Estimated fetal weight',
        'EFW centile',
        ],
        inplace=True
    )
    logger.debug('\tdimension, after dropping columns: {}'.format(df.shape))
    
    # drop columns with > 50% NA
    df.dropna(axis=1, thresh=int(df.shape[0]/2), inplace=True)
    logger.debug('\tdimension, after removing columns with more than 50% NA: {} '.format(df.shape))
    
    # drop rows which Fetus > 1 and drop the column
    df = df[df['Fetus'] == 1]
    df.drop(columns='Fetus', inplace=True)
    logger.debug('\tdimension, after removing >1 fetus: {}'.format(df.shape))
    
    # drop rows with NA
    df.dropna(axis=0, how='any', inplace=True)
    logger.debug('\tdimension, after removing rows with NA: {}'.format(df.shape))
    
    # rename columns
    df.index.names = ['no']
    df.columns = [
        'bpd',
        'hc',
        'ac',
        'fl',
        'efw', 
        'presentation', 
        'placenta_site', 
        'amniotic',
        'ga',
    ]
    
    # change index to int type
    df.index = df.index.astype(str)
    
    # combine some obvious categorical values
    df.loc[df['amniotic'] == 'increased', 'amniotic'] = 'polyhydramnios'
    df.loc[df['amniotic'] == 'reduced', 'amniotic'] = 'oligohydramnios'

    df.loc[df['presentation'].str.contains('breech', case=False), 'presentation'] = 'breech'
    df.loc[df['presentation'].str.contains('cephalic', case=False), 'presentation'] = 'cephalic'
    df.loc[df['presentation'] == 'oblique lie', 'presentation'] = 'other'
    df.loc[df['presentation'] == 'transverse', 'presentation'] = 'other'
    df.loc[df['presentation'] == 'Variable', 'presentation'] = 'other'

    df = df[df.placenta_site != 'right lateral posterior']
    df.loc[df['placenta_site'].str.contains('anterior', case=False), 'placenta_site'] = 'anterior'
    df.loc[df['placenta_site'] == 'placenta_site'] = 'posterior'
    df.loc[df['placenta_site'].str.contains('lateral', case=False), 'placenta_site'] = 'lateral'
    df.loc[df['placenta_site'].str.contains('placenta', case=False), 'placenta_site'] = 'previa'
    df.loc[df['placenta_site'].str.contains('posterior', case=False), 'placenta_site'] = 'posterior'

    #logger.debug(df.info())
    return df


def process_postnatal_data():
    logger.info('process postnatal data')
    df = pd.read_csv(postnatal_data, index_col=0)
    logger.debug('\tdimension, original df: {}'.format(df.shape))
    
    # drop all columns except the following
    df.drop(df.columns.difference([
        'Mother\'s Date of Birth (dd/mm/yy)',
        'Mother\'s Age at Delivery (Yrs)',
        'Mother Height (cm)',
        'Hypertension - Pregancy Induced or Essential [Nil = 0, PIH = 1, Essential HpT = 2]',
        'Diabetes - Gestational or Pregestational                                     [Nil=0, GDM=1, PRE-GDM=2]',
        'Date of Delivery (dd/mm/yy)',
        'Mode of Delivery [SVD=1, Forceps=2, Vacuum=3, LSCS=4,D&C=5, Breech delivery=6]',
    ]), axis=1, inplace=True)
    logger.debug('\tdimension, after dropping unused column: {}'.format(df.shape))
    
    # rename columns
    df.index.names = ['no']
    df.columns = [
        'mum_dob',
        'mum_age', 
        'mum_height', 
        'hypertension', 
        'diabetes', 
        'dod',
        'mod',
    ]
    
    # change index to int type
    df.index = df.index.astype(str)
    
    # clean up mum_dob column
    df.loc[df['mum_dob'] == '3/5/1986', 'mum_dob'] = '3-May-86'
    df.loc[df['mum_dob'] == '25/12/1986', 'mum_dob'] = '25-Dec-86'
    df.drop(df[df['mum_dob'] == '1-Jan-1290'].index, inplace=True)
    df.drop(df[df['mum_dob'] == '20-Apr-20'].index, inplace=True)
    df['mum_dob'] = pd.to_datetime(df['mum_dob'], format='%d-%b-%y')

    # clean up dod
    df.loc[df['dod'] == '6/1/2021', 'dod'] = '6-Jan-21'
    df['dod'] = pd.to_datetime(df['dod'], format='%d-%b-%y')

    # calculate mum_age at delivery
    df['mum_age'] = df['dod'] - df['mum_dob']
    df['mum_age'] = df['mum_age'] / np.timedelta64(1,'Y')

    # drop dob and dod after calculate age
    df.drop(columns=[
        'mum_dob', 
        'dod'],
        inplace=True
    )

    # drop rows with noise
    df.drop(df[df['mum_height'] == '*'].index, inplace=True)
    df = df[pd.to_numeric(df['mod'], errors='coerce').notnull()]

    # drop rows with NA
    df.dropna(axis=0, how='any', inplace=True)
    logger.debug('\tdimension, after removing rows with NA: {}'.format(df.shape))
    
    # regroup mode of delivery outcomes
    # SVD (1): 0
    # LSCS (4): 1
    # Others (2, 3, 5, 6): 2
    # sequence below is important, please don't swap!
    df['mod'] = df['mod'].astype(int)
    df.loc[df['mod'] == 1, 'mod'] = 0
    df.loc[df['mod'] == 2, 'mod'] = 2
    df.loc[df['mod'] == 3, 'mod'] = 2
    df.loc[df['mod'] == 4, 'mod'] = 1
    df.loc[df['mod'] == 5, 'mod'] = 2
    df.loc[df['mod'] == 6, 'mod'] = 2

    df.loc[df['hypertension'] == 0, 'hypertension'] = 'nil'
    df.loc[df['hypertension'] == 1, 'hypertension'] = 'pih'
    df.loc[df['hypertension'] == 2, 'hypertension'] = 'essential'
    df.loc[df['diabetes'] == 0, 'diabetes'] = 'nil'
    df.loc[df['diabetes'] == 1, 'diabetes'] = 'gdm'
    df.loc[df['diabetes'] == 2, 'diabetes'] = 'pregdm'

    #logger.debug(df.info())
    return df


####################
# merge ante and post data
####################

def merge_ante_post_data(df1, df2, onehot=False):
    logger.info('merge ante and post data')
    df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    df.index = df.index.astype(int)
    df = df.reset_index()
    
    if onehot:
        # convert categorical variables into dummy variables
        cat = ['presentation', 'placenta_site', 'amniotic', 'hypertension', 'diabetes']
        df = pd.get_dummies(df, columns=cat)
        df.filter(regex='_|'.join(cat) + '_').head()
    
    #logger.debug(df.info())
    logger.debug('\tnum of unique index: {}'.format(len(pd.unique(df['no']))))

    return df


####################
# chi square test
####################

def chi_square_test(df):
    logger.info('chi square test')
    features = ['presentation', 'placenta_site', 'amniotic', 'hypertension', 'diabetes']
    for feature in features:
        info = pd.crosstab(index=df[feature], columns=df['mod'])
        output = '\nChi Square Test between ' + feature + ' and MOD\n'
        output += str(info) + '\n'
        chi2, p, dof, expctd = chi2_contingency(info.values.tolist())
        output += 'chi2 value: {}\n'.format(chi2)
        output += 'p value: {}\n'.format(p)
        output += 'dof value: {}\n'.format(dof)
        output += 'expctd value: {}\n'.format(expctd)
        logger.debug(output)



####################
# feature selection
####################

def process_feature_selection(df, k=12, t='SelectKBest'):
    logger.info('process feature selection')
    selector = None
    if t == 'SelectKBest':
        selector = SelectKBest(f_classif, k=k)
    elif t == 'RFE':
        estimator = DecisionTreeClassifier()
        selector = RFE(estimator, n_features_to_select=k, step=1)

    X = df.drop(['no', 'mod'], axis=1)
    y = df['mod']
    selector.fit(X, y)
    print(selector.__dict__)
    cols = selector.get_support(indices=True)
    logger.debug('\tselected features: {}'.format(X.iloc[:, cols].columns.values))
    logger.debug('\tselected scores: {}'.format(selector.scores_))
    df = pd.concat([X.iloc[:, cols], df['mod']], axis=1)
    return df


####################
# get centile
####################

def LMS(input_df):
    stat_df = pd.DataFrame(columns = ['GA2', 'Count', 'Mean', 'Median', 'Max', 'Min', 'SD', 'Lambda', 'COV'])
    lms_df = pd.DataFrame(columns = ['GA2', 'P_2.5', 'P_3', 'P_5', 'P_10', 'P_25', 'P_50', 'P_75', 'P_90', 'P_95', 'P_97', 'P_97.5'])
    quantile_df = pd.DataFrame(columns = ['GA', 'P_2.5', 'P_3', 'P_5', 'P_10', 'P_25', 'P_50', 'P_75', 'P_90', 'P_95', 'P_97', 'P_97.5'])
    for x in range(int(min(input_df['ga'])), int(max(input_df['ga'])+1)):
        temp_df = input_df[(input_df['ga'] - x < 1) & (input_df['ga'] - x >= 0)].iloc[:, 1]#[input_df.columns[1]]
        temp_df = pd.to_numeric(temp_df)
        if len(temp_df) <= 1:
            continue
        #IQR = temp_df.quantile(.75) - temp_df.quantile(.25)
        #temp_df = temp_df[(temp_df < temp_df.mean() + IQR) & (temp_df > temp_df.mean() - IQR)]
        transformed_df, lambda_value = stats.boxcox(temp_df)
        stat_df.loc[len(stat_df)] = [x, 
                                 temp_df.size, 
                                 temp_df.mean(), 
                                 temp_df.median(),
                                 temp_df.max(),
                                 temp_df.min(),
                                 temp_df.std(),
                                 lambda_value, 
                                 temp_df.std() / temp_df.mean()                             
                                 ]
        lms_df.loc[len(lms_df)] = [x, 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * -1.960))**(1/stat_df[stat_df['GA2']==x]['Lambda'])), 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * -1.881))**(1/stat_df[stat_df['GA2']==x]['Lambda'])), 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * -1.645))**(1/stat_df[stat_df['GA2']==x]['Lambda'])), 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * -1.282))**(1/stat_df[stat_df['GA2']==x]['Lambda'])),  
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * -0.675))**(1/stat_df[stat_df['GA2']==x]['Lambda'])),  
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 0))**(1/stat_df[stat_df['GA2']==x]['Lambda'])),  
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 0.675))**(1/stat_df[stat_df['GA2']==x]['Lambda'])),  
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 1.282))**(1/stat_df[stat_df['GA2']==x]['Lambda'])),  
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 1.645))**(1/stat_df[stat_df['GA2']==x]['Lambda'])), 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 1.881))**(1/stat_df[stat_df['GA2']==x]['Lambda'])), 
                                     float(stat_df[stat_df['GA2']==x]['Median'] * (1 + (stat_df[stat_df['GA2']==x]['Lambda'] * stat_df[stat_df['GA2']==x]['COV'] * 1.960))**(1/stat_df[stat_df['GA2']==x]['Lambda']))
                                     ]
        quantile_df.loc[len(quantile_df)] = [x, 
                                   temp_df.quantile(.025), 
                                   temp_df.quantile(.03), 
                                   temp_df.quantile(.05),
                                   temp_df.quantile(.10),  
                                   temp_df.quantile(.25),  
                                   temp_df.quantile(.50),  
                                   temp_df.quantile(.75),  
                                   temp_df.quantile(.90),       
                                   temp_df.quantile(.95), 
                                   temp_df.quantile(.97), 
                                   temp_df.quantile(.975)
                                   ]
    return(stat_df, quantile_df, lms_df)


def nature_smooth_spline_stat(data):
    for f in ['Mean', 'Median', 'Max', 'Min']:
        for dof in [15, 16, 17, 18, 19, 20, 21, 22]:
            x = data['GA2']
            y = data[f]
            x_basis = cr(x, df=dof, constraints="center")
            y_hat = LinearRegression().fit(x_basis, y).predict(x_basis)
            data[(f + '_ss' + str(dof))] = y_hat
    return data


def get_stats_df(df):
    #Statistic
    #stat_df[x]
    statistic_df = [
        nature_smooth_spline_stat(LMS(df[(df['ga'] >= 11) & (df['ga'] <= 41)][['ga', 'bpd']].dropna())[0]), 
        nature_smooth_spline_stat(LMS(df[(df['ga'] >= 11) & (df['ga'] <= 41)][['ga', 'hc']].dropna())[0]), 
        nature_smooth_spline_stat(LMS(df[(df['ga'] >= 11) & (df['ga'] <= 41)][['ga', 'ac']].dropna())[0]), 
        nature_smooth_spline_stat(LMS(df[(df['ga'] >= 11) & (df['ga'] <= 41)][['ga', 'fl']].dropna())[0]), 
        nature_smooth_spline_stat(LMS(df[(df['ga'] >= 11) & (df['ga'] <= 41)][['ga', 'efw']].dropna())[0])
        ]
    for x in range(0, 5):
        for y in [15, 16, 17, 18, 19, 20, 21, 22]:
            statistic_df[x][('COV_ss' + str(y))] = statistic_df[x][('SD')] / statistic_df[x][('Mean_ss' + str(y))]

    return statistic_df


def cal_centile(GA, Type, Value, statistic_df):
    logger.debug('\tprocess centile for {} {} {}'.format(GA, Type, Value))
    type_df = statistic_df[Type]
    if int(GA) == 40:
        GA = 39
    type_df = type_df[type_df['GA2'] == int(GA)]
    #hack for 40
    return (stats.norm.sf(-((((Value / type_df['Median_ss18']) ** type_df['Lambda']) - 1) / (type_df['Lambda'] * type_df['COV_ss18']))) * 100)[0]


def get_centile(df):
    logger.info('get centile')
    statistic_df = get_stats_df(df)
    df['BPD_centile'] = np.nan
    df['HC_centile'] = np.nan
    df['AC_centile'] = np.nan
    df['FL_centile'] = np.nan
    df['EFW_centile'] = np.nan
    df['BPD_centile'] = df.apply(lambda x: cal_centile(x['ga'], 0, x['bpd'], statistic_df), axis=1)
    df['HC_centile'] = df.apply(lambda x: cal_centile(x['ga'], 1, x['hc'], statistic_df), axis=1)
    df['AC_centile'] = df.apply(lambda x: cal_centile(x['ga'], 2, x['ac'], statistic_df), axis=1)
    df['FL_centile'] = df.apply(lambda x: cal_centile(x['ga'], 3, x['fl'], statistic_df), axis=1)
    df['EFW_centile'] = df.apply(lambda x: cal_centile(x['ga'], 4, x['efw'], statistic_df), axis=1)
    #for x in range(len(df)):
    #    df.loc[x, 'BPD_centile'] = cal_centile(df['ga'][x], 0, df['bpd'][x])
    #    df.loc[x, 'HC_centile'] = cal_centile(df['ga'][x], 1, df['hc'][x])
    #    df.loc[x, 'AC_centile'] = cal_centile(df['ga'][x], 2, df['ac'][x])
    #    df.loc[x, 'FL_centile'] = cal_centile(df['ga'][x], 3, df['fl'][x])
    #    df.loc[x, 'EFW_centile'] = cal_centile(df['ga'][x], 4, df['efw'][x])
    
    return df


####################
# utilities
####################

def to_file(x, filename):
    with open(filename, 'wb') as f:
        pickle.dump(x, f)
    

def read_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.pipeline import make_pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import NearMiss, RandomUnderSampler, TomekLinks

from patsy import cr

from scipy.stats import chi2_contingency 

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import f_classif, RFE, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
    

from xgboost import XGBClassifier


SCALER = {
#    'no_scaler': None,
#    'standard_scaler': StandardScaler(),
#    'robust_scaler': RobustScaler(),
    'minmax_scaler': MinMaxScaler(),
    'maxabs_scaler': MaxAbsScaler(),
}

TOMEKLINKS = {
    'no_tomeklinks': None,
#    'tomeklinks': TomekLinks(),
}

SAMPLER = {
#    'no_sampler': None,
    'rus': RandomUnderSampler(random_state=88),
    'ros': RandomOverSampler(random_state=88),
    #'smote': SMOTENC(categorical_features=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], random_state=88),
#    'smote05': SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88, k_neighbors=5),
#    'smote10': SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88, k_neighbors=10),
    'smote25': SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88, k_neighbors=25),
    'nearmissv1': NearMiss(version=1),
#    'nearmissv1_05': NearMiss(version=1, n_neighbors=5),
#    'nearmissv1_10': NearMiss(version=1, n_neighbors=10),
#    'nearmissv1_25': NearMiss(version=1, n_neighbors=25),
#    'nearmissv2': NearMiss(version=2),
#    'nearmissv2_05': NearMiss(version=2, n_neighbors=5),
#    'nearmissv2_10': NearMiss(version=2, n_neighbors=10),
#    'nearmissv2_25': NearMiss(version=2, n_neighbors=25),
#    'nearmissv3': NearMiss(version=3),
#    'nearmissv3_05': NearMiss(version=3, n_neighbors_ver3=5),
#    'nearmissv3_10': NearMiss(version=3, n_neighbors_ver3=10),
#    'nearmissv3_25': NearMiss(version=3, n_neighbors_ver3=25),
}

ALGO = {
    'nb': GaussianNB(),
    'dt': DecisionTreeClassifier(random_state=88),
    'rf': RandomForestClassifier(random_state=88, n_jobs=-1),
    'knn': KNeighborsClassifier(n_jobs=-1),
    'lr': LogisticRegression(random_state=88, n_jobs=-1),
    'svm': SVC(random_state=88),
    'mlp': MLPClassifier(random_state=88),
    'xgb': XGBClassifier(random_state=88, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
}


####################
# machine learning
####################

# https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html
def machine_learning(X_train, X_test, y_train, y_test, scaler_func, tomeklinks_func, sampler_func, ml_func):
    steps = []
    if scaler_func is not None:
        steps.append(('scaler', scaler_func))
    if tomeklinks_func is not None:
        steps.append(('tomeklinks', tomeklinks_func))
    if sampler_func is not None:
        steps.append(('sampler', sampler_func))
    if ml_func is not None:
        steps.append(('estimator', ml_func))
        
    if sampler_func is None and tomeklinks_func is None:
        pipeline = Pipeline(steps)
    else:
        pipeline = imbPipeline(steps)
    #cv = StratifiedGroupKFold(n_splits=5, random_state=88, shuffle=True)
    cv = 5
    #score = cross_validate(pipeline, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1, groups=X_train['no'], verbose=0)
    score = cross_val_score(pipeline, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1, verbose=0)
    y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=-1, verbose=0)
    cm = confusion_matrix(y_train, y_pred)
    #print(cm)
    #print(score)

    return score.mean(), score.std(), (cm[0][0] / sum(cm[0])), (cm[1][1] / sum(cm[1])), (cm[2][2] / sum(cm[2]))
    #return score.mean(), score.std(), (cm[0][0] / sum(cm[0])), (cm[1][1] / sum(cm[1]))

#    total = 0.00
#    for train_index, val_index in cv.split(X_train, y_train, X_train['no']):
#        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
#        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
#
#        if scaler_func is not None:
#            scaler = scaler_func.fit(X_train_fold)
#            X_train_fold = scaler.transform(X_train_fold)
#            X_val_fold = scaler.transform(X_val_fold)
#        if tomeklinks_func is not None:
#            X_train_fold, y_train_fold = tomeklinks_func.fit_resample(X_train_fold, y_train_fold)
#        if sampler_func is not None:
#            X_train_fold, y_train_fold = sampler_func.fit_resample(X_train_fold, y_train_fold)
#
#        clf = ml_func.fit(X_train_fold, y_train_fold)
#        y_pred = clf.predict(X_val_fold)
#        score3 = balanced_accuracy_score(y_val_fold, y_pred)
#        total += score3
#    manual_score = total / 5
#
#    return score.mean(), score.std(), manual_score


def machine_learning_ensemble(X_train, X_test, y_train, y_test):
    models = [
        ('nb', make_pipeline(
            RobustScaler(), 
            NearMiss(version=3, n_neighbors_ver3=5), 
            GaussianNB()
        )),
        ('dt', make_pipeline(
            StandardScaler(), 
            TomekLinks(), 
            SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88, k_neighbors=25),
            DecisionTreeClassifier(random_state=88)
        )),
        ('rf', make_pipeline(
            StandardScaler(),
            TomekLinks(),
            SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88),
            RandomForestClassifier(random_state=88)
        )),
        ('knn', make_pipeline(
            MinMaxScaler(),
            NearMiss(version=2, n_neighbors=10),
            KNeighborsClassifier()
        )),
        ('lr', make_pipeline(
            MaxAbsScaler(),
            SMOTENC(categorical_features=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], random_state=88, k_neighbors=25),
            LogisticRegression(random_state=88)
        )),
        ('svm', make_pipeline(
            MaxAbsScaler(), 
            TomekLinks(), 
            RandomUnderSampler(random_state=88), SVC(random_state=88)
        ))
    ]
    ensemble = VotingClassifier(estimators=models, voting='hard')
    #cv = StratifiedGroupKFold(n_splits=5, random_state=88, shuffle=True)
    cv = 5
    #score = cross_validate(ensemble, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1, groups=X_train['no'], verbose=0)
    score = cross_val_score(ensemble, X_train, y_train, scoring='balanced_accuracy', cv=cv, n_jobs=-1, verbose=0)
    y_pred = cross_val_predict(ensemble, X_train, y_train, cv=cv, n_jobs=-1, verbose=0)
    cm = confusion_matrix(y_train, y_pred)
    #print(cm)
    #print(score)

    return score.mean(), score.std(), (cm[0][0] / sum(cm[0])), (cm[1][1] / sum(cm[1])), (cm[2][2] / sum(cm[2]))
    #return score.mean(), score.std(), (cm[0][0] / sum(cm[0])), (cm[1][1] / sum(cm[1]))


def process_machine_learning(df):
    logger.info('process machine learning')
    total = len(SCALER) * len(TOMEKLINKS) * len(SAMPLER) * len(ALGO)
    i = 0
    results = []
    X_train, X_test, y_train, y_test = train_test_split(df.drop('mod', axis=1), df['mod'], test_size=0.15, random_state=88)
    for k,v in SCALER.items():
        for k2,v2 in TOMEKLINKS.items():
            for k3,v3 in SAMPLER.items():
                for k4,v4 in ALGO.items():
                    result = {}
                    i += 1
                    #result['ba'], result['std'], result['c0'], result['c1'], result['c2'] = machine_learning(X_train, X_test, y_train, y_test, v, v2, v3, v4)
                    result['ba'], result['std'], result['c0'], result['c1'], result['c2'] = machine_learning(df.drop('mod', axis=1), None, df['mod'], None, v, v2, v3, v4)
                    logger.debug('\t%d/%d: process [%s][%s][%s][%s]\t[%.02f%%][%f][%.02f%%][%.02f%%][%.02f%%]' % (i, total, k, k2, k3, k4, result['ba'] * 100, result['std'], result['c0'] * 100, result['c1'] * 100, result['c2'] * 100))
                    #result['ba'], result['std'], result['c0'], result['c1']  = machine_learning(df.drop('mod', axis=1), None, df['mod'], None, v, v2, v3, v4)
                    #logger.debug('\t%d/%d: process [%s][%s][%s][%s]\t[%.02f%%][%f][%.02f%%][%.02f%%]' % (i, total, k, k2, k3, k4, result['ba'] * 100, result['std'], result['c0'] * 100, result['c1'] * 100))
                    result['scaler'] = k
                    result['sampler'] = k2
                    result['tomeklinks'] = k3
                    result['algorithm'] = k4
                    results.append(result)
    return results


def process_machine_learning_ensemble(df):
    logger.info('process machine learning ensemble')
    X_train, X_test, y_train, y_test = train_test_split(df.drop('mod', axis=1), df['mod'], test_size=0.15, random_state=88)
    results = []
    result = {}
    #result['ba'], result['std'], result['c0'], result['c1'], result['c2'] = machine_learning_ensemble(X_train, X_test, y_train, y_test)
    result['ba'], result['std'], result['c0'], result['c1'], result['c2'] = machine_learning_ensemble(df.drop('mod', axis=1), None, df['mod'], None)
    logger.debug('\tprocess ensemble\t[%.02f%%][%f][%.02f%%][%.02f%%][%.02f%%]' % (result['ba'] * 100, result['std'], result['c0'] * 100, result['c1'] * 100, result['c2'] * 100))
    #result['ba'], result['std'], result['c0'], result['c1'] = machine_learning_ensemble(df.drop('mod', axis=1), None, df['mod'], None)
    #logger.debug('\tprocess ensemble\t[%.02f%%][%f][%.02f%%][%.02f%%]' % (result['ba'] * 100, result['std'], result['c0'] * 100, result['c1'] * 100))
    results.append(result)
    return results


####################
# result 
####################

def process_result(result):
    logger.info('process result')
    df = pd.DataFrame.from_dict(result)
    return df


####################
# main
####################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store_true', help='preprocess')
    parser.add_argument('-b', action='store_true', help='machine learning')
    parser.add_argument('-c', action='store_true', help='result')
    args = parser.parse_args()

    is_run_all = True if len(sys.argv) == 1 else False

    file_pd = '1_df_pd.csv'
    file_ml = '2_bin_ml.csv'
    file_result = '3_df_result.csv'

    if is_run_all or args.a:

        # preprocess
        df1 = process_antenatal_data()
        df2 = process_postnatal_data()
        df = merge_ante_post_data(df1, df2, onehot=True)
        #df = merge_ante_post_data(df1, df2, onehot=False)
        #chi_square_test(df)
        #df = process_feature_selection(df, 10, t='RFE')
        df = process_feature_selection(df, 12)
        #df = get_centile(df)
        df.to_csv(file_pd)

    if is_run_all or args.b:
        # machine learning
        df = pd.read_csv(file_pd, index_col=0)
        #print(df)
        #results = process_machine_learning(df)
        results = process_machine_learning_ensemble(df)
        to_file(results, file_ml)

    if is_run_all or args.c:
        # result
        results = read_file(file_ml)
        df_results = process_result(results)
        df_results.to_csv(file_result)

