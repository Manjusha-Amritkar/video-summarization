# Title: Reconstruction-Aware Multi-Head Attention for Unsupervised Video Summarization
This repository is useful for summarizing the video into short, crisp summary of longer video contents. 
# Dataset
TVSum Dataset
Location: https://www.kaggle.com/datasets/andreymasyutin/tvsum-dataset

About: TV Sum has 50 videos divided into 10 semantic categories (e.g., news, sports, cooking), taken from YouTube. Each video is given importance scores by 20 users, from 1 to 5.

SumMe Dataset
Location: https://zenodo.org/records/4884870?utm_source

About: SumMe is a collection of 25 videos created by users to show different real-life situations. Each video has been annotated by several users who provide ground truth summaries for the evaluation. The average length of the videos is around 2 minutes.
# Usage Instructions
Clone the repository, download the dataset from the given link, install all dependecies. You will be able to execute the code

# Requirements


# Methodology
It uses hybrid architecture of Bi-LSTM and Multi head attention model along with reconstruction error to train the model. It is fully unsupervised model. It identifies keyframes as per reconstruction difficulty. 
