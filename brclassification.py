import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc)

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords


# Stopwords

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list


def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
    
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def preprocess_text(text):
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)

    return text


def main():
    project = 'caffe' # select one of the cvs
    path = f'datasets/{project}.csv'

    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state= 999) # Shuffle

    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

    
    datafile = 'Title+Body.csv'
    out_csv_name = f'./{project}_NB.csv'

    data = pd.read_csv(datafile).fillna('')
    text_col = 'text'

    original_data = data.copy

    # Text cleaning
    data[text_col] = data[text_col].apply(preprocess_text)

    iterations = 30

    # Metrics

    accuracies =[]
    precisions = []
    recalls = []
    f1_scores = []

    params = {
        'alpha': [0.1, 0.5, 1.0, 1.5],
        'fit_prior': [True, False]
    }

    for repeated_time in range(iterations):
        indices = np.arange(data.shape[0])
        train_index, test_index = train_test_split(
            indices, test_size=0.2, random_state=repeated_time
        )

        train_text = data[text_col].iloc[train_index]
        test_text = data[text_col].iloc[test_index]

        y_train = data['sentiment'].iloc[train_index]
        y_test  = data['sentiment'].iloc[test_index]

        tfidf = TfidfVectorizer(
            ngram_range = (1, 2),
            max_features = 1000
        )
        X_train = tfidf.fit_transform(train_text)
        X_test = tfidf.transform(test_text)


        clf = MultinomialNB()

        grid = GridSearchCV(
            clf,
            params,
            cv=5,
            scoring='roc_auc'
        )
        grid.fit(X_train, y_train)

        best_clf = grid.best_estimator_
        best_clf.fit(X_train, y_train)

        y_pred = best_clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)  

        prec = precision_score(y_test, y_pred, average='macro')
        precisions.append(prec)

        rec = recall_score(y_test, y_pred, average='macro')
        recalls.append(rec)

        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
    
    final_accuracy  = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall    = np.mean(recalls)
    final_f1        = np.mean(f1_scores)

    print(f"Average Accuracy:      {final_accuracy:.4f}")
    print(f"Average Precision:     {final_precision:.4f}")
    print(f"Average Recall:        {final_recall:.4f}")
    print(f"Average F1 score:      {final_f1:.4f}")

    try:
        # Attempt to check if the file already has a header
        existing_data = pd.read_csv(out_csv_name, nrows=1)
        header_needed = False
    except:
        header_needed = True

    df_log = pd.DataFrame(
        {
            'repeated_times': [iterations],
            'Accuracy': [final_accuracy],
            'Precision': [final_precision],
            'Recall': [final_recall],
            'F1': [final_f1],
        }
    )

    df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

    print(f"\nResults have been saved to: {out_csv_name}")


if __name__ == "__main__":
    main()
