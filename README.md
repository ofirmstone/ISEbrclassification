<div align="center">
  
# Bug Report Classification - Tool Building Project

<em>Identifying performance-related bugs using machine learning</em>

<img src="https://img.shields.io/github/last-commit/ofirmstone/bug-report-classification?style=flat&logo=git&logoColor=white&color=blue" alt="last-commit">

<em>Built with the following tools and technologies:</em>

<img alt="Static Badge" src="https://img.shields.io/badge/Python-%233776AB?logo=python&logoColor=white">
<img alt="Static Badge" src="https://img.shields.io/badge/Pandas-%23150458?logo=pandas&logoColor=white">
<img alt="Static Badge" src="https://img.shields.io/badge/NumPy-%23013243?logo=numpy&logoColor=white">
<img alt="Static Badge" src="https://img.shields.io/badge/scikit--learn-%23F7931E?logo=scikitlearn&logoColor=white">

</div>

---

## Overview

In my coursework, I discussed the problems of bug reports for large projects (such as in deep-learning like TensorFlow), and how machine learning techniques can assist the process of bug report classification and triage. I then built an intelligent tool for bug report classification that classifies whether a bug report is performance related or not.

My research, proposed solution, experiments and observations can be found in the **ISE_Task1.pdf** file within this repository.

## Dependencies
- Python 3.6+
- Required packages:
  ```bash
  pandas
  numpy
  scikit-learn
  nltk
  ```

## How to replicate the test results

- Change the project name between provided dataset names when testing
  ```bash
  project = 'caffe' # select one of the cvs
  path = f'datasets/{project}.csv'
  ```
- Run the code (terminal command below)
  ```bash
  python brclassification.py
  ```
- Record the results from either the outputted csv file or console
