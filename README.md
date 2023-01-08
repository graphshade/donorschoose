# Predicting Exciting Donation Projects

<img src="https://github.com/graphshade/donorschoose/blob/main/results/Historical%20distribution%20of%20exciting%20versus%20non-exciting%20projects.png" />

<h2>Problem statement</h2>
The current annual rate of donation project proposals that end-up fully funded at Donor's Choose is about 15%. That means, 75 out of every 100 projects are not fully funded and deemed not exciting. The organization's mission is to help teachers fund projects they need, and if 75% of those projects end up not fully funded, is Donor's Choose achieving its mission? The chart above, showing the historical distribution of exciting projects versus nonexciting projects, visually illustrates the problem the organization faces. 

There is a need for management of Donor's Choose to understand what attributes make certain projects exciting and what attributes make donors repeat donors. 

<br></br>
To help the organization, I attempt to build a model to predict whether a new project proposal is exciting or not exciting. Lastly, I used descriptive analysis and clustering to understand what characteristics make certain donors repeat donors.

<h2>Languages and Libraries Used</h2>

- Python
- [List of libraries](https://github.com/graphshade/donorschoose/blob/main/requirements.txt)

<h2>Environment Used </h2>

- <b>Ubuntu 22.04.1 LTS</b>


<h2>Key Findings:</h2>

1. Exciting projects have at least 25% of great comments on the projects page
2. Exciting projects have least one non-teacher referred donor, aand at least one teacher-referred donor
3. The estimated cost of the project including tip has marginal effect on amount donated


<h2>Recommendations:</h2>

Based on the findings, the following recommendations were made:
1. Operate the model at the 50% predictive probability of default threshold. At this level, the selected model can identify 63% of loan default cases. Historically, this could saved $3 million in loan defaults.This saving in loan default is computed without taking into consideration the cost of associated with wrongly classifying a non-default loan as default. Specifically, operating at the 50% threshold results into wrongly classing 9 out of every 100 observations

2. The customer service department may use the predicted probability of default to segment borrowers and device an outreach campaign targeted at the various segments. For instance, for customers with high probability of default, the customer service department may constantly follow up to understand their financial situation and help them restructure their payment plan

<h2>Reproducing the Analysis:</h2>

<p align="left">

1. [Install R and RStudio](https://techvidvan.com/tutorials/install-r/)
 
2. Clone the project: Run this from the command line
 
 ```commandline
 git clone https://github.com/graphshade/loan_default.git
 ```
 
3. Install Required Libraries Using Virtual Environment: 
   
   You may install the libraries directly on your computer however, using the virtual environment library `renv`. [Follow this guide to install renv](https://www.youtube.com/watch?v=yc7ZB4F_dc0)
   1. Open the app.R file in RStudio
   2. In the RStudio console run `renv::init()` to initiate the renv virtual environment and install the required libraries from the [renv.lock](https://github.com/graphshade/loan_default/blob/main/renv.lock) file 
