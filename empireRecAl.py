import base64
from PIL import Image
from collections import Counter
from io import BytesIO
from statistics import mean
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split,LeaveOneOut,KFold,learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,roc_curve, auc
import psycopg2,os
import seaborn as sns
import sys
from sklearn.metrics import confusion_matrix
from joblib import dump,load
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from flask import Flask, render_template
from datetime import datetime

OPPORTUNITIES_PATH = './opportunites.csv'
OPPORTUNITIES_FAILED_PATH = './init_failed.csv'
OPPORTUNITIES_NORMALIZED_PATH = './test_opportunites_normalized.csv'
RATING_FILE_PATH = './rating.csv'
FILE_REPORT_NAME = './report.txt'
MODEL_FILE_PATH = './modelRandomForest.joblib'

MOST_SIMILAR_OPPORTUNITY = 5
MOST_SIMILAR_USERS = 5
K_FOLDS = 5
N_FREQUENCY_MEMBERS = 3


os.environ["LOKY_MAX_CPU_COUNT"] = "2"
app = Flask(__name__,template_folder=os.path.abspath("templates"))

scaler = MinMaxScaler()

# Connect to your postgres DB
conn = psycopg2.connect(database="empire", user="epuzzle_api_development", password="dMCgEonVijn7wIu", host="epuzzle-psql-database-development.internal", port=5432)

# Open a cursor to perform database operations
cur = conn.cursor()


def query_from_DB(query):

    # Execute a query
    cur.execute(query)

    # Retrieve query results
    records = cur.fetchall()
    
    return records


columns_to_normalize = {
    'risk_level': lambda x: 5 - x,  # Inverter risco (níveis mais baixos são melhores)
    'distribution_commission': lambda x: -x,
    'sale_commission': lambda x: -x,  
    'acquisition_commissions': lambda x: -x,  
    'taxes': lambda x: -x,  
    'licensing_fees': lambda x: -x,  
    'other_management_costs': lambda x: -x,  
    'other_financial_costs': lambda x: -x,  
    'development_taxes': lambda x: -x,  
    'promotion_costs': lambda x: -x,  
    'formalities': lambda x: -x,  
    'investment_duration_months': lambda x: 24 - x,
    'market_value': lambda x: x,
    'income_from_sale': lambda x: x,
    'project_yield': lambda x: x / 100,  # Se for percentagem, converte para [0,1]
}


# scaling_columns = ["asking_price","total_investment","risk_level","distribution_commission","acquisition_commissions","sale_commission","taxes","licensing_fees","other_management_costs","other_financial_costs","rents","promoter_benefit","development_taxes","income_from_sale","promotion_costs","market_value","deed_value","formalities",'deed_proportion','taxes_proportion', 'formalities_proportion','acquisition_commissions_proportion','distribution_commission_proportion','licensing_fees_proportion','promotion_costs_proportion','other_financial_costs_proportion','net_revenue_distribuiton','project_yield']

scaling_columns = ["risk_level","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","total_investment","deed_proportion","taxes_proportion","formalities_proportion","acquisition_commissions_proportion","distribution_commission_proportion","licensing_fees_proportion","promotion_costs_proportion","other_financial_costs_proportion","net_revenue_distribuiton",'project_yield']

def RandomForest_Report(model,X_train,y_train,y_test,y_pred,report,accuracy,kf_mean_accuracy,kf_report,prob_accept,prob_reject,predicted_class,arrayMostEqual,TITLE_REPORT):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(FILE_REPORT_NAME, "a", encoding="utf-8") as f:
        f.write("=============================================\n")
        f.write(f"📅 NEW EXECUTION: {now}\n")
        f.write(""+TITLE_REPORT+"\n\n")

        f.write("🔁 LEAVE-ONE-OUT VALIDATION:\n")
        f.write(f" Total oportunidades avaliadas: {len(y_test)}\n")
        f.write(f" 📌 MEAN ACCURACY: {round(accuracy, 3)}\n")
        f.write(f" 📌 PRECISION (classe 1): {round(report['1']['precision'], 3)}\n")
        f.write(f" 📌 RECALL (classe 1): {round(report['1']['recall'], 3)}\n")
        f.write(f" 📌 F1-SCORE (classe 1): {round(report['1']['f1-score'], 3)}\n\n")
        print(f"📌 ACCURACY: {accuracy:.4f}\n")

        # f.write("📊 K-FOLD CROSS VALIDATION:\n")
        # f.write(f" - FOLD USED: {K_FOLDS}\n")
        # f.write(f" - MEAN ACCURACY: {round(kf_mean_accuracy, 3)}\n")
        # f.write(f" - PRECISION (classe 1): {round(kf_report['1']['precision'], 3)}\n")
        # f.write(f" - RECALL (classe 1): {round(kf_report['1']['recall'], 3)}\n")
        # f.write(f" - F1-SCORE (classe 1): {round(kf_report['1']['f1-score'], 3)}\n\n")
        
        img_base64_conf = conf_matrix(y_test,y_pred)
        # metrics_graph(report,kf_report)

        img_base64_features = variables_weight(X_train,model)

        # f.write("📋CLASSIFICATION REPORT:")
        # report_df = pd.DataFrame(report).transpose()
        # f.write(report_df+"\n")

        f.write("TRAIN DISTRIBUITON:")
        f.write(str(np.bincount(y_train))+'\n')
        f.write("TEST DISTRIBUITON:")
        f.write(str(np.bincount(y_test))+'\n') 

        f.write(f"\n🔮 NEW opportunity PREDICT\n")
        f.write(f"✅ PROB. BEING ACCEPTED: {prob_accept:.2f}\n")
        f.write(f"❌ PROB. BEING REJECTED: {prob_reject:.2f}\n")
        f.write(f"🟢 **FINAL RESULT:** {predicted_class}\n")

        f.write("📋 5 MOST SIMILAR OPPORTUNITIES:")

        for i in range(len(arrayMostEqual)):
            f.write(" - "+str(arrayMostEqual.iloc[i])+"\n")
        
        f.write("=============================================\n\n")

    return img_base64_conf,0 #img_base64_features

def log_write(content):

    f = open("log.txt","w")
    f.write(content)

def conf_matrix(y_test,y_pred):

    conf_mat = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Accepted"], yticklabels=["Rejected", "Accepted"])
    plt.xlabel("Predict")
    plt.ylabel("Real")
    plt.title("ConfusionMatrix")
    plt.show()

    
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return img_base64

def metrics_graph(report,kf_report):

    loo_precision = report['weighted avg']['precision']
    loo_recall = report['weighted avg']['recall']
    loo_f1 = report['weighted avg']['f1-score']

    kf_precision = kf_report['weighted avg']['precision']
    kf_recall = kf_report['weighted avg']['recall']
    kf_f1 = kf_report['weighted avg']['f1-score']

    methods = ['LOOCV', 'K-Fold']
    precision = [loo_precision, kf_precision]
    recall = [loo_recall, kf_recall]
    f1_score = [loo_f1, kf_f1]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(methods, precision, marker='o', label='Precision', color='blue')
    plt.plot(methods, recall, marker='s', label='Recall', color='green')
    plt.plot(methods, f1_score, marker='^', label='F1 Score', color='red')

    plt.title('Comparison of Evaluation Metrics: LOOCV vs K-Fold')
    plt.xlabel('Validation Method')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./graph", bbox_inches="tight")


def variables_weight(X_train,model):
    # Obter importância das features
    feature_importances = model.feature_importances_

    # Criar um DataFrame para melhor visualização
    feature_names = X_train.columns  # Se estiveres a usar um DataFrame Pandas
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Ordenar pela importância
    importances_df = importances_df.sort_values(by="Importance", ascending=False)

    # Printar o ranking das features
    print(importances_df)

    # Gráfico de barras das features mais importantes
    plt.figure(figsize=(10, 6))
    plt.barh(importances_df["Feature"], importances_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance on Random Forest")
    plt.gca().invert_yaxis()  # Inverter para mostrar a mais importante no topo
    plt.savefig("./importance.png", bbox_inches="tight")

    
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()

    return img_base64

def listMostEqual(similarity,data):

    similarity = similarity[-1][:-1] 
    top_n_indices = np.argsort(similarity)[::-1][:MOST_SIMILAR_OPPORTUNITY]  # Pegamos os 5 mais parecidos

    return data.iloc[top_n_indices][["opportunities_id","name"]]


def loo_RandomForest(similarity_matrix, data,original,loadModel):
    X = similarity_matrix[:-1]  # Todas as linhas menos a última (nova oportunidade)
    
    # X = data.drop(['accepted'], axis=1)  # Remove a coluna primeiro
    # X = X[:-1]  # Depois remove a última linha
    y = data["accepted"][:-1]  # y também tem de remover a última linha

    loo = LeaveOneOut()  
    model=0

    if (loadModel == 0):
        model = RandomForestClassifier(random_state=42)  
    else:
        model = load(MODEL_FILE_PATH)

    y_true,y_pred,accuracies = [],[],[]  # Lista para armazenar os resultados
    precisions, recalls, f1_scores,y_prob = [], [], [],[]


    for train_index, test_index in loo.split(X):
        # x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        # y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        x_train, x_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]

        smote = SMOTE(random_state=42)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

        # model.fit(x_train,y_train)
        model.fit(x_train_resampled, y_train_resampled)
        pred = model.predict(x_test)

        y_true.append(y_test.iloc[0])
        y_pred.append(pred[0])
        accuracies.append(accuracy_score([y_test.iloc[0]], pred))

        proba = model.predict_proba(x_test)[0][1]
        y_prob.append(proba)
        precisions.append(precision_score([y_test.iloc[0]], pred, zero_division=0))
        recalls.append(recall_score([y_test.iloc[0]], pred, zero_division=0))
        f1_scores.append(f1_score([y_test.iloc[0]], pred, zero_division=0))

    mean_accuracy = sum(accuracies) / len(accuracies)  
    report = classification_report(y_true,y_pred,output_dict=True)

    # === K-Fold Cross Validation ===
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    kf_accuracies = []
    kf_y_true, kf_y_pred = [], []

    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        kf_accuracies.append(accuracy_score(y_test, pred))
        kf_y_true.extend(y_test)
        kf_y_pred.extend(pred)

    kf_mean_accuracy = sum(kf_accuracies) / len(kf_accuracies)
    kf_report = classification_report(kf_y_true, kf_y_pred, output_dict=True)

    new_opportunity = similarity_matrix[-1].reshape(1, -1)
    new_opportunity = new_opportunity.reshape(1, -1)
    # new_array = X.iloc[-1].to_numpy()
    # new_opportunity = new_array.reshape(1, -1)

    # fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # roc_auc = auc(fpr, tpr)

    # plt.figure()
    # plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='blue')
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve - Leave-One-Out')
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # # img_base64_conf,img_base64_features = [],[]

    # train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42)

    # train_scores_mean = np.mean(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)

    # plt.figure()
    # plt.plot(train_sizes, train_scores_mean, label="Training score", marker='o', color='blue')
    # plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", marker='o', color='lightblue')
    # plt.title("Learning Curve - Random Forest")
    # plt.xlabel("Training Set Size")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # # Plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(accuracies, label='Accuracy', marker='o')
    # plt.plot(precisions, label='Precision', marker='x')
    # plt.plot(recalls, label='Recall', marker='s')
    # plt.plot(f1_scores, label='F1 Score', marker='^')
    # plt.xlabel('LOO Iteration')
    # plt.ylabel('Score')
    # plt.title('Metric Evolution per Leave-One-Out Iteration')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    probabilities = model.predict_proba(new_opportunity)
    prob_accept = probabilities[0][1]  # Probabilidade de ser aceite
    prob_reject = probabilities[0][0]  # Probabilidade de ser rejeitado
    predicted_class = "ACCEPTED" if prob_accept > prob_reject else "REJECTED"

    arrayMostEqual = listMostEqual(similarity_matrix,original)
    # img_base64_conf,img_base64_features = [],[]
    
    img_base64_conf,img_base64_features = RandomForest_Report(model,x_train,y_train,y_true,y_pred,report,mean_accuracy,kf_mean_accuracy,kf_report,prob_accept,prob_reject,predicted_class,arrayMostEqual,"🔍 TESTS REPORT - RANDOM FOREST WITH LEAVE-ONE-OUT\n")

    if(loadModel == 0):
        dump(model,MODEL_FILE_PATH)

    return predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features

def RandomForest(similarity_matrix,data,original,loadModel):
    X = similarity_matrix[:-1]  # Todas as linhas menos a última (nova oportunidade)
    # X = data.drop(['accepted'], axis=1)  # Remove a coluna primeiro
    # X = X[:-1]  # Depois remove a última linha
    y = data["accepted"][:-1]  # y também tem de remover a última linha

    model = 0 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # oversample = SMOTE(sampling_strategy='auto',random_state=42)
    # over_X_train, over_y_train = oversample.fit_resample(X_train, y_train)
    
    if (loadModel == 0):
        model = RandomForestClassifier(random_state=42)  
    else:
        model = load("modelRandomForest.joblib")

    model.fit(X_train, y_train)
    # model.fit(over_X_train, over_y_train)
    
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test,y_pred,output_dict=True)
    

    # Prever a nova oportunidade
    new_opportunity = similarity_matrix[-1].reshape(1, -1)
    new_opportunity = new_opportunity.reshape(1, -1)
    # new_array = X.iloc[-1].to_numpy()
    # new_opportunity = new_array.reshape(1, -1)

    probabilities = model.predict_proba(new_opportunity)
    prob_accept = probabilities[0][1]  # Probabilidade de ser aceite
    prob_reject = probabilities[0][0]  # Probabilidade de ser rejeitado
    predicted_class = "ACCEPTED" if prob_accept > prob_reject else "REJECTED"

    arrayMostEqual = listMostEqual(similarity_matrix,original)

    img_base64_conf,img_base64_features = [],[]

    if(loadModel == 0):
        dump(model,'modelRandomForest.joblib')

    img_base64_conf,img_base64_features = RandomForest_Report(model,X_train,y_train,y_test,y_pred,report,accuracy,prob_accept,prob_reject,predicted_class,arrayMostEqual,"🔍 TESTS REPORT - SIMPLE RANDOM FOREST\n")

    return predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features

def recommend_opportunity(input_opportunitie,data,loadModel):

    dataset = pd.concat([data,input_opportunitie],ignore_index=True)
    
    dataset_prepared = data_vars_prepare(dataset)

    data_normalized = normalize_data(dataset_prepared)

    data_normalized.to_csv("normalized.csv")

    similarity_matrix = cosine_similarity(data_normalized.drop(["accepted"],axis=1))

    # predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features = RandomForest(similarity_matrix,data_normalized,dataset,load)
    
    predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features = loo_RandomForest(similarity_matrix,data_normalized,data,loadModel)

    return predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features

def normalize_data(data):

    # Aplicar as transformações
    for col, transform in columns_to_normalize.items():
        if col in data:
            data[col] = data[col].apply(transform)

    data[scaling_columns] = scaler.fit_transform(data[scaling_columns])

    return data

def calculate_rating(row):
    # Pesos para cada fator na fórmula de rating
    risk_weight = 0.25
    net_revenue_distribuiton_weight = 0.15
    project_weight = 0.30
    year_weight = 0.15
    duration_weight = 0.15
    
    # Calcular scores com valores normalizados
    risk_score = float(row['risk_level']) * risk_weight
    income_score = (float(row['net_revenue_distribuiton'])) * net_revenue_distribuiton_weight
    cost_score = float(row['project_yield']) * project_weight
    duration_score = float(row['investment_duration_months']) * duration_weight
    year_score = float(row['building_year']) * year_weight

    # Rating final: soma dos scores, limitado entre 0 e 1
    rating = max(0, min(1, risk_score + income_score + cost_score + duration_score + year_score))
    
    return rating

def data_vars_prepare(data):
        
    investment_columns = ['deed_value','taxes', 'formalities','acquisition_commissions','distribution_commission','licensing_fees','promotion_costs','other_financial_costs'] 

    # Repita o processo para outras colunas
    columns_to_convert = ["deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission"]

    for col in columns_to_convert:
        data[col] = data[col].fillna(0)  # Substituir NaN por 0
        data[col] = data[col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)  # Converter para inteiro

    data = data.fillna(0)

    data['total_investment'] = data[investment_columns].sum(axis=1)

    data['deed_proportion'] = (data['deed_value']/data['total_investment'])*100
    data['taxes_proportion'] = (data['taxes']/data['total_investment'])*100
    data['formalities_proportion'] = (data['formalities']/data['total_investment'])*100
    data['acquisition_commissions_proportion'] = (data['acquisition_commissions']/data['total_investment'])*100
    data['distribution_commission_proportion'] = (data['distribution_commission']/data['total_investment'])*100
    data['licensing_fees_proportion'] = (data['licensing_fees']/data['total_investment'])*100
    data['promotion_costs_proportion'] = (data['promotion_costs']/data['total_investment'])*100
    data['other_financial_costs_proportion'] = (data['other_financial_costs']/data['total_investment'])*100

    if(str(data['management_fee']) in '2018'):
        data['net_revenue_distribuiton'] = 0.1*((data['income_from_sale']+data['rents'])-(data['sale_commission']-data['total_investment']-data['promoter_benefit']-data['development_taxes']-data['other_management_costs']-data['promoter_benefit']))
    else:
        data['net_revenue_distribuiton'] = 0.123*((data['income_from_sale']+data['rents'])-(data['sale_commission']-data['total_investment']-data['promoter_benefit']-data['development_taxes']-data['other_management_costs']-data['promoter_benefit']))

    data['project_yield'] = round((data['net_revenue_distribuiton']/data['total_investment'])*100,2)
    
    data['rating'] = (data.apply(calculate_rating, axis=1)*5).round(1)

    # Criar a coluna 'Accepted': 1 se status NÃO for 'Archived', 0 caso contrário
    data["accepted"] = data["status"].apply(lambda x: 0 if x == "ARCHIVED" else 1)

    data = data.drop(['opportunities_id','name','fractions','property_category','address_country','management_fee','status'],axis=1)

    data = pd.get_dummies(data)
    
    # data = data.fillna(0)

    return data

def init_data(opportunity):
    oportunities_accepted = query_from_DB("SELECT o.id,o.name,o.address_country,o.risk_level,o.fractions,o.building_year,o.property_category,o.market_segment,o.investment_duration_months,o.market_value,o.asking_price,o.deed_value,o.taxes,o.formalities,o.development_taxes,o.acquisition_commissions,o.distribution_commission,o.licensing_fees,o.promotion_costs,o.other_management_costs,o.income_from_sale,o.other_financial_costs,o.rents,o.promoter_benefit,o.sale_commission,o.management_fee,o.status FROM opportunities o where o.id !='"+opportunity+"'")
    
    data = pd.DataFrame(oportunities_accepted,columns=["opportunities_id","name","address_country","risk_level","fractions","building_year","property_category","market_segment","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","management_fee","status"])
    
    failed_data = pd.read_csv(OPPORTUNITIES_FAILED_PATH).drop(['investment_value','yield','motivo'],axis=1)

    data = pd.concat([data,failed_data],ignore_index=True)

    return data

def input_opp(opportunity):

    input = query_from_DB("SELECT o.id,o.name,o.address_country,o.risk_level,o.fractions,o.building_year,o.property_category,o.market_segment,o.investment_duration_months,o.market_value,o.asking_price,o.deed_value,o.taxes,o.formalities,o.development_taxes,o.acquisition_commissions,o.distribution_commission,o.licensing_fees,o.promotion_costs,o.other_management_costs,o.income_from_sale,o.other_financial_costs,o.rents,o.promoter_benefit,o.sale_commission,o.management_fee,o.status FROM opportunities o where o.id ='"+opportunity+"'")

    input_data = pd.DataFrame(input,columns=["opportunities_id","name","address_country","risk_level","fractions","building_year","property_category","market_segment","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","management_fee","status"])

    return input_data

# -------------------------------------------------------------------------- NEW USER RECOMMENDATION 'USER-USER' -------------------------------------------------------------------


def list_most_equal_users(similarity,data,index):

    similarityIndex = similarity[index]

    # Criamos uma máscara para excluir o próprio índice
    similarityIndex[index] = -np.inf  # Definimos um valor muito baixo para evitar a seleção do próprio usuário

    # Selecionamos os 5 índices com maior similaridade
    top_n_indices = np.argsort(similarityIndex)[::-1][:MOST_SIMILAR_USERS]

    return data.iloc[top_n_indices]

def vars_user_prepare(listMembers):

    listMembers_tosimilarity = listMembers.drop(['Member_ID','Name'],axis=1)

    print(listMembers_tosimilarity)
    listMembers_tosimilarity['BIRTHDATE'] = (listMembers_tosimilarity['BIRTHDATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

    listMembers_tosimilarity = listMembers_tosimilarity.fillna(0)

    listMembers_tosimilarity = pd.get_dummies(listMembers_tosimilarity)

    listMembers_tosimilarity = listMembers_tosimilarity.astype({col: int for col in listMembers_tosimilarity.select_dtypes(include=['bool']).columns})

    return listMembers_tosimilarity

def search_for_investments_for_user(listMostSimilarUsers): 

    listMembers = query_from_DB("SELECT o.id FROM opportunities o, members m,level2 l,investments i WHERE l.id = i.level2_fk AND m.id = l.member_fk AND l.member_fk IN('"+listMostSimilarUsers.iloc[0]+"','"+listMostSimilarUsers.iloc[1]+"','"+listMostSimilarUsers.iloc[2]+"','"+listMostSimilarUsers.iloc[3]+"','"+listMostSimilarUsers.iloc[4]+"') AND o.id = i.opportunity_fk GROUP BY o.id")
    
    data = pd.DataFrame(listMembers,columns=["OPPORTUNITY_ID"])

    return data

def list_member_similar_invesment(opportunity_id):

    data = []
    if(not pd.isna(opportunity_id)):
        listMembers = query_from_DB("SELECT l.member_fk, CONCAT(m.first_name,' ',m.last_name) as Name FROM investments i, level2 l, members m WHERE i.opportunity_fk='"+opportunity_id+"' AND i.level2_fk = l.id AND l.member_fk = m.id")
        data = pd.DataFrame(listMembers,columns=["Member_ID","Name"])

    return data

def count_frequency_members(membersList): 

    count = membersList.value_counts()
        
    return count[count >= N_FREQUENCY_MEMBERS].index.tolist()

def search_new_users_recomm(newFlag):
    listMembersToInvest = []

    listMembers = query_from_DB("SELECT m.id, CONCAT(m.first_name, ' ', m.last_name) AS Name,l.delivery_city,l.billing_city,l.billing_country,l.civil_status,l.gender,l.birthdate,COUNT(i.id) FROM members m,level2 l,investments i WHERE l.id = i.level2_fk AND m.id = l.member_fk GROUP BY m.id,l.delivery_city,l.billing_city,l.billing_country,l.civil_status,l.gender,l.birthdate")
    
    data = pd.DataFrame(listMembers,columns=["Member_ID","Name","DELIVERY_CITY","BILLING_CITY","BILLING_COUNTRY","CIVIL_STATUS","GENDER","BIRTHDATE","COUNT"])

    # Para teste porque nao ha users sem investimentos
    new_member = [["0c5c155e-1111-111e-826b-005051220568", "Goncalo Correia","Coimbra","Coimbra","PT","SINGLE","MALE", pd.Timestamp("2000-05-12 00:30:00"),"0"]] 
    aux = pd.DataFrame(new_member,columns=["Member_ID","Name","DELIVERY_CITY","BILLING_CITY","BILLING_COUNTRY","CIVIL_STATUS","GENDER","BIRTHDATE","COUNT"])
    data = pd.concat([data,aux], ignore_index=True)

    if(newFlag == 1):
        listMembersToInvest = data.loc[data['COUNT'] == '0']
    else:
        listMembersToInvest = data.loc[data['COUNT'] != '0']

    return data,listMembersToInvest

def users_recommendation_algorithm(newFlag):

    listMembers, listMembersToInvest = search_new_users_recomm(newFlag)

    print(listMembers)

    listMembers_tosimilarity = vars_user_prepare(listMembers)

    print(listMembers_tosimilarity)

    similarity_matrix = cosine_similarity(listMembers_tosimilarity)

    print(similarity_matrix)

    data_to_recommend = {}  
    
    for i in range(len(listMembersToInvest)):

        member_index = listMembers[listMembers['Member_ID'].str.contains(listMembersToInvest.iloc[i]["Member_ID"], case=False, na=False)].index[0]

        listofMostEqualUsers = list_most_equal_users(similarity_matrix,listMembers,member_index)

        data_to_recommend[listMembersToInvest.iloc[i]["Member_ID"]] = search_for_investments_for_user(listofMostEqualUsers["Member_ID"]).to_numpy()

    return data_to_recommend

# -------------------------------------------------------------------------- USUAL USER RECOMMENDATION 'USER-OPP' -------------------------------------------------------------------

def loo_RandomForest_USER(data,data_to_predict,original,loadModel):
    
    data = data.fillna(0)
    X = data.drop(['accepted'], axis=1)
    X = X[:-1]  # Remove a coluna primeiro
    # X = X.drop(index=6)  # Remove a coluna primeiro
    y = data["accepted"][:-1]

    data_to_predict = data_to_predict.drop(['accepted'], axis=1)  # Remove a coluna primeiro
    
    loo = LeaveOneOut()  
    model=0

    if (loadModel == 0):
        model = RandomForestClassifier(random_state=42)  
    else:
        model = load(MODEL_FILE_PATH)

    y_true,y_pred,accuracies,y_prob,precisions,recalls,f1_scores   = [],[],[],[],[],[],[]  # Lista para armazenar os resultados


    for train_index, test_index in loo.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # x_train, x_test = X[train_index], X[test_index] 
        # y_train, y_test = y[train_index], y[test_index]

        class_counts = Counter(y_train)
        min_class = min(class_counts, key=class_counts.get)
        min_count = class_counts[min_class]
        safe_k = max(1, min(min_count - 1, 5))  # Nunca inferior a 1


        smote = SMOTE(random_state=42,k_neighbors=safe_k)  # Nunca inferior a 1)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

        # model.fit(x_train,y_train)
        model.fit(x_train_resampled, y_train_resampled)
        pred = model.predict(x_test)

        y_true.append(y_test.iloc[0])
        y_pred.append(pred[0])
        # Adiciona a probabilidade da classe positiva (ACEITE = 1)
        proba = model.predict_proba(x_test)[0][1]
        y_prob.append(proba)

        accuracies.append(accuracy_score([y_test.iloc[0]], pred))
        precisions.append(precision_score([y_test.iloc[0]], pred, zero_division=0))
        recalls.append(recall_score([y_test.iloc[0]], pred, zero_division=0))
        f1_scores.append(f1_score([y_test.iloc[0]], pred, zero_division=0))

    mean_accuracy = sum(accuracies) / len(accuracies)  
    report = classification_report(y_true,y_pred,output_dict=True)


    # # === K-Fold Cross Validation ===
    # kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    # kf_accuracies = []
    # kf_y_true, kf_y_pred = [], []

    # for train_index, test_index in kf.split(X):
    #     x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #     # x_train, x_test = X[train_index], X[test_index]
    #     # y_train, y_test = y[train_index], y[test_index]

    #     model.fit(x_train, y_train)
    #     pred = model.predict(x_test)

    #     kf_accuracies.append(accuracy_score(y_test, pred))
    #     kf_y_true.extend(y_test)
    #     kf_y_pred.extend(pred)

    # kf_mean_accuracy = sum(kf_accuracies) / len(kf_accuracies)
    # kf_report = classification_report(kf_y_true, kf_y_pred, output_dict=True)

    # new_opportunity = similarity_matrix[-1].reshape(1, -1)
    # new_opportunity = new_opportunity.reshape(1, -1)

    # Prever a nova oportunidade
    # new_opportunity = similarity_matrix[-1].reshape(1, -1)
    # new_opportunity = new_opportunity.reshape(1, -1)
    new_array = X.iloc[-1].to_numpy()
    new_opportunity = new_array.reshape(1, -1)

    probabilities = model.predict_proba(new_opportunity)
    prob_accept = probabilities[0][1]  # Probabilidade de ser aceite
    prob_reject = probabilities[0][0]  # Probabilidade de ser rejeitado
    predicted_class = "ACCEPTED" if prob_accept > prob_reject else "REJECTED"

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Leave-One-Out')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # img_base64_conf,img_base64_features = [],[]

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score", marker='o', color='blue')
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", marker='o', color='lightblue')
    plt.title("Learning Curve - Random Forest")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(accuracies, label='Accuracy', marker='o')
    plt.plot(precisions, label='Precision', marker='x')
    plt.plot(recalls, label='Recall', marker='s')
    plt.plot(f1_scores, label='F1 Score', marker='^')
    plt.xlabel('LOO Iteration')
    plt.ylabel('Score')
    plt.title('Metric Evolution per Leave-One-Out Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    img_base64_conf,img_base64_features = RandomForest_Report(model,x_train,y_train,y_true,y_pred,report,mean_accuracy,0,0,prob_accept,prob_reject,predicted_class,[],"🔍 RELATÓRIO DE TESTES - RANDOM FOREST COM LEAVE-ONE-OUT\n")

    if(loadModel == 0):
        dump(model,MODEL_FILE_PATH)

    return predicted_class,[],prob_accept,prob_reject,img_base64_conf,img_base64_features

def search_history_investment_user(member_id): 

    oportunities_ofMember = query_from_DB("SELECT o.id,o.name,o.address_country,o.risk_level,o.fractions,o.building_year,o.property_category,o.market_segment,o.investment_duration_months,o.market_value,o.asking_price,o.deed_value,o.taxes,o.formalities,o.development_taxes,o.acquisition_commissions,o.distribution_commission,o.licensing_fees,o.promotion_costs,o.other_management_costs,o.income_from_sale,o.other_financial_costs,o.rents,o.promoter_benefit,o.sale_commission,o.management_fee,o.status FROM opportunities o, members m,level2 l,investments i WHERE l.member_fk='"+member_id+"' AND m.id = l.member_fk AND l.id = i.level2_fk AND o.id = i.opportunity_fk GROUP BY o.id")

    data = pd.DataFrame(oportunities_ofMember,columns=["opportunities_id","name","address_country","risk_level","fractions","building_year","property_category","market_segment","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","management_fee","status"])
    
    not_oportunities_ofMember = query_from_DB("SELECT o.id,o.name,o.address_country,o.risk_level,o.fractions,o.building_year,o.property_category,o.market_segment,o.investment_duration_months,o.market_value,o.asking_price,o.deed_value,o.taxes,o.formalities,o.development_taxes,o.acquisition_commissions,o.distribution_commission,o.licensing_fees,o.promotion_costs,o.other_management_costs,o.income_from_sale,o.other_financial_costs,o.rents,o.promoter_benefit,o.sale_commission,o.management_fee,o.status FROM opportunities o, members m,level2 l,investments i WHERE l.member_fk!='"+member_id+"' AND m.id = l.member_fk AND l.id = i.level2_fk AND o.id = i.opportunity_fk AND o.status='ONGOING' GROUP BY o.id")

    data_to_predict = pd.DataFrame(not_oportunities_ofMember,columns=["opportunities_id","name","address_country","risk_level","fractions","building_year","property_category","market_segment","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","management_fee","status"])
    
    return data,data_to_predict

def user_recommend_with_history(member_id):

    history_investment_user,data_to_predict = search_history_investment_user(member_id)

    dataset_prepared = data_vars_prepare(history_investment_user)

    dataset_prepared["accepted"] = 1

    data_normalized = normalize_data(dataset_prepared)
    
    print(data_to_predict["opportunities_id"])
    dataset_to_predict_prepared = data_vars_prepare(data_to_predict)
    
    dataset_prepared["accepted"] = 0

    data_to_predict_normalized = normalize_data(dataset_to_predict_prepared)

    data = pd.concat([data_normalized,data_to_predict_normalized],ignore_index=True)

    predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features = loo_RandomForest_USER(data,data_to_predict_normalized,history_investment_user,0)

    # return predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features


    return history_investment_user

def calculate_diversity_score(opportunities_df, selected_indices):
    similarity_matrix = cosine_similarity(opportunities_df)
    selected_sim = similarity_matrix[np.ix_(selected_indices, selected_indices)].copy()
    np.fill_diagonal(selected_sim, np.nan)
    mean_similarity = np.nanmean(selected_sim)

    return 1 - mean_similarity

def select_most_diverse(opportunities_df, k=5):
    similarity_matrix = cosine_similarity(opportunities_df)
    diversity_matrix = 1 - similarity_matrix
    selected = [0]
    for _ in range(k - 1):
        remaining = list(set(range(len(similarity_matrix))) - set(selected))
        next_idx = max(
            remaining,
            key=lambda i: min([diversity_matrix[i][j] for j in selected])
        )
        selected.append(next_idx)
    
    # Cria o heatmap com limite de cores ajustado ao range minúsculo
    sns.heatmap(diversity_matrix, cmap='Blues', vmin=0, vmax=0.000001,
                square=True, linewidths=0.5, cbar_kws={'label': 'Diversity'})

    plt.title("Diversity Heatmap Between Opportunities")
    plt.xlabel("Opportunity Index")
    plt.ylabel("Opportunity Index")
    plt.tight_layout()
    plt.show()


    return selected

def plot_diversity_comparison(opportunities_df, top_similar_indices,diverse_indices):

    diversity_similar = calculate_diversity_score(opportunities_df, top_similar_indices)
    diversity_diverse = calculate_diversity_score(opportunities_df, diverse_indices)

    labels = ['Top-5 Most Similar', 'Top-5 Most Diverse']
    scores = [diversity_similar, diversity_diverse]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=scores)
    plt.ylabel("Diversity Score")
    plt.title("Diversity Comparison Between Recommendation Strategies")
    plt.ylim(0, 0.0000015)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    
    return diversity_similar,diversity_diverse

def metrics_cold_start(oportunities):

    oportunities_comp = query_from_DB("SELECT o.id,o.name,o.address_country,o.risk_level,o.fractions,o.building_year,o.property_category,o.market_segment,o.investment_duration_months,o.market_value,o.asking_price,o.deed_value,o.taxes,o.formalities,o.development_taxes,o.acquisition_commissions,o.distribution_commission,o.licensing_fees,o.promotion_costs,o.other_management_costs,o.income_from_sale,o.other_financial_costs,o.rents,o.promoter_benefit,o.sale_commission,o.management_fee,o.status FROM opportunities o where o.id ='"+oportunities[0]+"'"+"OR o.id ='"+oportunities[1]+"'"+"OR o.id ='"+oportunities[2]+"'"+"OR o.id ='"+oportunities[3]+"'"+"OR o.id ='"+oportunities[4]+"'"+"OR o.id ='"+oportunities[5]+"'"+"OR o.id ='"+oportunities[6]+"'")
    
    data = pd.DataFrame(oportunities_comp,columns=["opportunities_id","name","address_country","risk_level","fractions","building_year","property_category","market_segment","investment_duration_months","market_value","asking_price","deed_value","taxes","formalities","development_taxes","acquisition_commissions","distribution_commission","licensing_fees","promotion_costs","other_management_costs","income_from_sale","other_financial_costs","rents","promoter_benefit","sale_commission","management_fee","status"])
    
    dataset_prepared = data_vars_prepare(data)
    data_normalized = normalize_data(dataset_prepared)

    # Selecionar índices top-k (ex: 5 mais similares)
    top_k_indices = list(range(5))  # ou outra lógica

    # Calcular as mais diversas
    diverse_indices = select_most_diverse(data_normalized)

    # Mostrar gráfico comparativo
    diversity_similar,diversity_diverse = plot_diversity_comparison(data_normalized, top_k_indices, diverse_indices)

    # Calcular diversity dos top_k (apenas para referência)
    print(f"Diversity Score (Top-k similar): {diversity_similar:.10f}")
    print(top_k_indices)


@app.route("/")
def main ():
    id = ''
    array_answers = []
    array_members = []
    members2Recommend = []
    opportunitie_id = query_from_DB("SELECT o.id,o.status FROM opportunities o")
    data = pd.DataFrame(opportunitie_id,columns=["opportunities_id","status"])
    data["accepted"] = data["status"].apply(lambda x: 0 if x == "ARCHIVED" else 1)
    status ='❌'

    option = 0 

    print("-------------------------- EMPIRE PUZZLE RECOMMENDATION ALGORITHM --------------------------")
    print("--------------------------           1 - TRAIN ALGORTHIM          --------------------------")
    print("--------------------------           2 - NEW USER RECOMMEND       --------------------------")
    print("--------------------------           3 - USER RECOMMEND           --------------------------")
    print("--------------------------           4 - NEW OPP MAY LIKE         --------------------------")
    print("--------------------------           5 - LEAVE                    --------------------------")
    print("-------------------------- EMPIRE PUZZLE RECOMMENDATION ALGORITHM --------------------------")
    option = input("OPTION: ")



    if(option == '1'):

        # for i in range(len(data)):

        id = str(data.iloc[4]["opportunities_id"])
        dataset = init_data(id)
        new_opportunitie = input_opp(id)

        predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features = recommend_opportunity(new_opportunitie,dataset,0)
            
        #     if(predicted_class == 'ACCEPTED' and data.iloc[i]["accepted"] == 1):
        #         status = '✅'

        #     array_answers.append([id,data.iloc[i]["accepted"],status])

        # answers = pd.DataFrame(array_answers,columns=["OPPORTUNITIES_ID","VALUE","GUESSED"])
        # print(answers)
        option = 0

    elif(option == '2'):
        data_to_recommend = users_recommendation_algorithm(1)
        print(data_to_recommend)

        metrics_cold_start(list(data_to_recommend.values())[0].flatten())
        
        option = 0
    elif(option == '3'):

        member_id = "6e738dba-3d94-4a27-878e-6041d56e073c"
        data_to_recommend = user_recommend_with_history(member_id)

        option = 0

    elif(option == '4') :
        
        id = str(data.iloc[12]["opportunities_id"])
        print(id)
        dataset = init_data(id)
        new_opportunitie = input_opp(id)

        predicted_class,arrayMostEqual,prob_accept,prob_reject,img_base64_conf,img_base64_features = recommend_opportunity(new_opportunitie,dataset,1)

        if(predicted_class == 'ACCEPTED' and data.iloc[15]["accepted"] == 1):
            status = '✅'
            membersList = pd.DataFrame([],columns=["Member_ID","Name"])   
            for j in range(len(arrayMostEqual)):
                aux_list = list_member_similar_invesment(arrayMostEqual.iloc[j]["opportunities_id"])
                membersList = pd.concat([membersList,aux_list],ignore_index=True)
                
            members2Recommend = count_frequency_members(membersList)

        array_members = pd.DataFrame(members2Recommend,columns=["Member_ID","Name"])
        
        array_answers.append([id,data.iloc[5]["accepted"],status])

        option = 0


        arrayMostEqual = arrayMostEqual.to_dict(orient='records')
        array_members = array_members.to_dict(orient='records')

    img = Image.open("./importance.png")
    buffer = BytesIO()
    img.save(buffer, format="PNG") 
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return render_template('index.html',id=id,prob_accept = prob_accept,prob_reject = prob_reject,arrayMostEqual=arrayMostEqual,array_members=array_members,var_image=img_base64)




if __name__ == "__main__":
    app.run(debug=True, extra_files=[os.path.join("templates", f) for f in os.listdir("templates")])
    # main()