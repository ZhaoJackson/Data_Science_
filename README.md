# Data Science – Learning-in-Progress

Welcome! This repository documents an ongoing learning journey through Elements of Data Science. The work is active and evolving over time. Collaboration from the Columbia University Data Science community is welcome—issues, ideas, and pull requests are appreciated.

- **Status**: in progress; notebooks and materials updated periodically
- **Focus**: Python data stack, data wrangling, visualization, statistics, ML, NLP, clustering, and more
- **Collaboration**: Open to contributions from Columbia Data Science Department peers and the broader community

---

## Weekly Modules (What each week covers)
Below are the canonical topics based on the notebooks in this repo. Consider renaming week folders to these clearer titles for discoverability.

- Week 1 – Introduction to Data Science Tools (`week1/week1.ipynb`)
  - Python DS stack, conda/Anaconda, Jupyter/IPython, terminal basics
- Week 2 – Python Review, NumPy, and Pandas (`week2/week2.ipynb`)
  - NumPy arrays, indexing, aggregations; Pandas Series/DataFrames, selection, missing data
- Week 3 – Pandas, Data Exploration, and Visualization (`weekly_quiz/Week_03_Quiz-zz3119.ipynb`, `week3/week3.ipynb` if present)
  - EDA with Pandas; plotting with Matplotlib/Seaborn; data vocabulary
- Week 4 – Confidence Intervals and Hypothesis Testing (`week4/week4.ipynb`)
  - Sampling, CIs, permutation tests, A/B testing, p-values, MAB
- Week 5 – Introduction to Machine Learning Models (`week5/week5.ipynb`)
  - Supervised/unsupervised learning overview; linear models; classification vs regression
- Week 6 – ML Models (kNN, Trees, Ensembles, OVR) (`week6/eods-week06-intro_to_machine_learning_models_continued.ipynb`)
  - kNN, Decision Trees, Ensembles (bagging/boosting/stacking), multiclass OVR
- Week 7 – Model Evaluation and Hyperparameter Tuning (`week7/eods-week07-model_evaluation_and_hyperparameter_tuning.ipynb`)
  - Train/test splits, metrics (MSE/RMSE, R², accuracy, precision/recall/F1, ROC-AUC), regularization, tuning
- Week 9 – Joining Data and Dimensionality Reduction (`week9/eods-week09-joining_data_and_dimensionality_reduction.ipynb`)
  - Joins/merges; feature selection (LASSO, tree importance, RFE); PCA
- Week 10 – NLP, Sentiment Analysis, and Topic Modeling (`week 10/eods-week10-nlp_sentiment_analysis_and_topic_modeling.ipynb`)
  - Pipelines; text features; sentiment; topic modeling overview
- Week 11 – Clustering, Recommendation Systems, Imbalanced Classes (`week 11/eods-week11-clustering_recommendationsystems_imbalancedclasses.ipynb`)
  - k-Means and clustering concepts; recommender systems; handling class imbalance

Note: Some weeks (e.g., 8) are covered via larger notebooks (see `week 8/`), and quizzes/homeworks also reinforce topics.

---

## Repository Structure
- `week1/ ... week11/` – Weekly lecture notebooks and examples
- `Final/` – Final exam prep materials and exercises (notebooks and docs)
- `Mid-Term/` – Midterm notebooks and study materials
- `homeworks/` – HW folders with datasets and notebooks
- `weekly_quiz/` – Weekly quiz notebooks
- `data/` – Example datasets used across notebooks (e.g., `wine_dataset.csv`, `yellowcab_*.csv`, `BicycleWeather.csv`)
- `src/` – Small Python scripts and toy Flask examples
- `images/` – Supporting images for notebooks

Key datasets referenced:
- `data/wine_dataset.csv` – ML classification/regression demos (Weeks 5–7)
- `data/yellowcab_demo_withdaycategories.csv` – CI/hypothesis testing and sampling (Week 4)
- `data/FremontBridge_2012-2015.csv`, `data/BicycleWeather.csv` – time series examples

---

## Suggested Folder Renames (optional)
To improve readability, consider renaming the week folders (keeping original in parentheses until fully migrated):
- `week1/` → `01-intro-to-data-science-tools/`
- `week2/` → `02-python-numpy-pandas/`
- `week3/` → `03-pandas-eda-visualization/`
- `week4/` → `04-confidence-intervals-hypothesis-testing/`
- `week5/` → `05-intro-to-ml-models/`
- `week6/` → `06-ml-models-continued-knn-trees-ensembles/`
- `week7/` → `07-model-evaluation-and-hyperparameter-tuning/`
- `week8/` → `08-data-cleaning-and-feature-engineering/`
- `week9/` → `09-joining-data-and-dimensionality-reduction/`
- `week 10/` → `10-nlp-sentiment-topic-modeling/`
- `week 11/` → `11-clustering-recsys-imbalanced-classes/`

If you adopt these names, update links inside notebooks as needed.

---

## Environment Setup
- Recommended: `conda` environment using `requirements.yml`
- Start Jupyter: `jupyter lab` (or `jupyter notebook`)
- Python 3.9–3.10 compatible; core libs include NumPy, Pandas, Matplotlib, Seaborn, scikit-learn; some notebooks use `mlxtend`

---

## Contributing
Contributions are welcome!
- Open an issue to propose improvements or new learning materials
- Submit PRs for: fixes, clearer naming, additional examples, or new week summaries
- Columbia Data Science Department peers: feel free to add reading lists, suggested problems, or guest lecture notes

Please keep changes modular and include short explanations in PR descriptions.

---

## Acknowledgements
- Course inspiration: Elements of Data Science from professor Andi Cupallari
- References: Python Data Science Handbook, Python Machine Learning, Data Science From Scratch
