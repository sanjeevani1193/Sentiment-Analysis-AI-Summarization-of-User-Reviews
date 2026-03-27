# Sentiment-Analysis-and-AI-Summarization-of-User-Reviews

In this project, I have analyzed  a large corpus of Amazon product reviews to extract sentiment and generate AI-powered summaries of the provided user feedback. It demonstrates a complete data analysis workflow: data cleaning, EDA, sentiment classification (VADER), comparison with provided ratings(score), AI-based summarization (Hugging Face Transformers) of helpful reviews, and interactive visualizations which provide insights for decision-makers.

## Motivation

Product teams rely on user feedback to make data-driven decisions. This project allows them to make data-driven decisions after: -
- Performing sentiment analysis on user‑generated text.
- Validating sentiment against provided score/ratings.
- Identifying the most helpful reviews (based on community votes) and summarize their content using AI.
- Providing visual insights to support product teams in understanding user pain points and strengths.

## Dataset

- **Source:** [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) (Kaggle)
- **Size:** 568,454 reviews
- **Columns used:** `Text` (review), `Score` (star rating), `HelpfulnessNumerator/Denominator` (helpfulness votes)

## Methodology

1. **Data Cleaning and Preprocessing**  
   - Dropped missing values and duplicates  
   - Created a `clean_text` column (lowercase, letters only) for VADER  
   - Derived `score_sentiment` from score/ratings (1-2=negative, 3=neutral, 4-5=positive)  
   - Computed `helpfulness_ratio` = Numerator / (Denominator+1e-6) and flagged `is_helpful` (ratio ≥ 0.8)

2. **Sentiment Analysis with VADER**  
   - Applied VADER to `clean_text` to get compund scores and sentiment labels (positive/negative/neutral) under `vader_sentiment`  
   - Compared VADER with score‑based sentiment to measure agreement (agreement **~80%**)

3. **AI Summarization**  
   - Filtered for **helpful** reviews (`is_helpful=True`)  
   - Sampled 100 helpful positive and 100 helpful negative reviews  
   - Used Hugging Face `sshleifer/distilbart-cnn-12-6` summarization pipeline to generate concise summaries for each group

4. **Visualizations**  
   - Interactive plots (Plotly) for:-
     - Sentiment distribution
     - Box plot (ratings vs sentiment)
     - Agreement heatmap
     - Helpfulness histograms 
   - Word clouds for positive and negative reviews

## Key Findings and Observations

- **Overall Sentiment:** 87.8% of reviews are positive, 9.88% are negative, and 2.31% are neutral.
- **Agreement with Score Ratings:** VADER and score ratings agree on ~80% of reviews. The disagreements could be due to the reviews containing a mixed sentiment or sarcasm.
- **Helpfulness Patterns:** 34.8% of reviews had a helpfulness ratio ≥ 0.8 (i.e., ~35% of users found the reviews helpful). Helpful reviews were more likely to be negative; this informs us which sentiment type was most valued by the community.
- **VADER vs Score:** The box plot shows that reviews labeled negative by VADER have a median rating of 2, while positive VADER reviews have a median rating of 4. This aligns with expectations and validates VADER’s performance. However, there are some negative‑sentiment reviews still received high star ratings, often due to mixed feedback which pose as the outliers.

## Visualizations

All interactive HTML dashboards are available in the `visualizations/` folder.

## How to Reproduce

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) and place `Reviews.csv` in the `data/` folder.
4. Run the Jupyter notebook `analysis.ipynb`.

**Note:** The full dataset is large; the notebook loads it in chunks. For quick testing, you can limit rows with `nrows=10000`.

## Tools & Libraries

- Python (pandas, numpy, re)
- VADER for sentiment analysis
- Hugging Face Transformers (distilbart-cnn-12-6)
- Plotly, Matplotlib, WordCloud
- Jupyter Notebook

## Future Work

- Experiment with more advanced sentiment models (e.g., BERT)
- Incorporate topic modeling to automatically group themes
- Deploy as a Streamlit app for interactive exploration
- Integrate time‑series analysis to track sentiment trends over time.

## Author

**Sanjeevani Rajpurohit** – [GitHub](https://github.com/sanjeevani1193)
