# E-Commerce_Recommender_System (Tianchi Competition)

# Problem Statement
A recommendation system is a tool that uses a series of algorithms, analyze the customer logs and make recommendations online. Nowdays online business competition is server but with opportunities. A good online recommendation system can bring company tremendous profits and provide customers customized services. Netflix is a world known company boosted in few years, which is an example of the power of recommendation system. People who is good at working with recommendation system takes the key positions in company reserach department. In this case, we are targeting the online shopping data in 2014 from Alibaba taobao and trying to build a recommendation system with algorithms to improve the customers experineces and logistic processes. We will also try to optimize the models with decision tree and neural network. 

# Contents

| files | description |
| ---| --- |
| datasets | user.csv, and item.csv |
| Readme| Read me including details on this project |
| notebooks | EDA, feature extraction, baseline prediction, collaborative filter, LR, GBDT, WDL |
| model | LR, GBDT, WDL, K-means, NFM py files, tf_model  |
| feature construction | extracted part 1, part 2, part 3 features  |

* tianchi_fresh_comp_train_item.csv, tianchi_fresh_comp_train_user.csv datasets download from https://tianchi.aliyun.com/competition/entrance/231522/introduction


# Workflow
1. EDA and baseline prediction
  a. Download datasets
  b. EDA on average order rate, user behavior analysis
  c. Merge features based on rule(customer order in 24h) and make prediction as baseline

2. Feature engineer and extraction
  a. user, item, category features and feature intereaction
  b. time decay, 6 day in week as train, 1 day label prediciton 
  c. period time, target three weeks as 3 parts 

3. LR & GDBT model
  a. K-means cluster 
  b. train, val data
  c. LR & GDBT

4. Wide/Deep model
  a. WDL
  b. NFM


# Executive Summary
Tianchi_fresh_comp_train_item.csv, tianchi_fresh_comp_train_user.csv datasets were downloaded from Tianchi Alicloud. user.csv 1GB contains 30M row of customers oders and columns of user_id, item_id, behavior_type, user_geohash, tem_category and time. We ignored user_geohas incomplete data. tianchi_fresh_comp_train_item.csv contains the columns of item_id, item_geohash, item_category of item information. Through the EDA, I found the customer order rate was 0.009985776926023916, which is extreme unbalanced data. The customers order behaviors different from others with different browser, order rate, stay time values. I extracted the feature with behavior 3(cart), behavior 4(order) and found the customers would place order within 24h of place cart. So, based on this rule, I used 2014-12-18 user cart items to predict next days order items. F1 score was used to evalute the model competency. The first try ended up with a F1 score 0.66 in Tianchi Competitation ranking 256. 

Next, feature engineer was conducted to the datasets. The datasets carries lots of information in the user, item, time and the principle was to extract as much features as possible. So one-hot encoding, embedding, discreet continuous values were included in labeling. The e-commerce was sensitive to times so time was endetailed into different time periods. I merged columns intersted from datasets to explore the feature intereaction importance. The datasets then were divided into three parts, part 1 and part 2 inlucded train and validation data, part 3 only included train dataset and used for prediction. Each part covered a 6 days as training data, last day as prediction. Three datasets also explained the repreatability. This is a binary classification problem and the possibilities of orders were the target. 

Collaborative filter did not included because the less sparsity of this data. I firstly used logistic regression to classify the labels. In the EDA section, I found the unbalanced data(p/n, 1:120). To avoid failure in training, I did subsampling on the negative samples. To avoid the insufficent feature sampling, K-means was used before subsample to guarantee subsample covered balanced datasets. Missing values were filled with -1. NP ratios were evaluated in Logistic Regression and NP ratio 55 ended with lowerst f1 score. Then LR model predicted the values, which were submitted. This results turned to a f1 score of 0.05982, indicating LR did not work well on this problem. There could be more than linear components in this datasets. 

GBDT is a better model to this case: 1. this is a binary classification problem. 2. GDBT model updated each residual during learning processes and leaned to more accurate. 3. no-linear model. We evaluated n/p ratios with f1 score under GBDT, along with the hyperparameters: min_samples_leaf 10, max_depth 4,10,learning rate 0.025, n_estimators 300, subsmaple 0.8.GDBT gave us results with a 0.1044. The significant improvement indicated the fittness of randomness, non-linear model。More hyperparameters fine tuning could improve more, which is not included in this project. Cross Entropy Error Function was used as loss function in both LR and GDBT. 

I further passed the data with more deep models, wide deep network model, based on the publications from Google on recommendation system. Wide deep model was used for matching. We passed the contineuous variables to concatenate layers with embedded categorical variables. For example, the item category was processed with embedding by hush buckets as deep columns. Then passed to linear wide model. The categorical variables were through cross product transformation and used log loss function. In this case, the category sparse was limited to wide deep model, so we only get a f1 score of 0.00278. The datasets were lack of sparsity for WDL model. 
To the lecture's notes: NFM would not fit for this problem for the same reasons of less spasity, I only included the NFM model code in the notebook for reference.

# Conclusions
1. GBDT model is reliable to work on this type of data on recommendation system.
2. Feature engineer and feature extraction should carry business insight regarding overall data.
3. More work on hyperparameters of GBDT to improve the score and ranking.
4. Multiple models can be built for the recommendation system.
5. Attentions could be helpful to build RCNN model for recommendation system.
