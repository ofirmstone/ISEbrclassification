# Bug Report Classification - ISE Coursework Tool Building Project

In my coursework, I discussed the problems of bug reports for large projects (such as in deep-learning like TensorFlow), and how machine learning techniques can assist the process of bug report classification and triage. My research, proposed solution, experiments and observations can be found in the **ISE_Task1.pdf** file within this repository.

# Requirements
- Python 3.6+
- Required packages:
```bash
pandas
numpy
scikit-learn
nltk
```

# How to Run the Code (Manual)

```bash
python brclassification.py
```

# How to replicate the results

- Change project name between provided dataset names when testing:
  ```bash
  project = 'caffe' # select one of the cvs
  path = f'datasets/{project}.csv'
  ```
- Run the code
- Record the results from the outputted csv file or console
