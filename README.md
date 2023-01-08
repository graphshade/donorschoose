# Predicting Exciting Donation Projects

<img src="https://github.com/graphshade/donorschoose/blob/main/results/Historical%20distribution%20of%20exciting%20versus%20non-exciting%20projects.png" />

<h2>Problem statement</h2>
The current annual rate of donation project proposals that end-up fully funded at Donor's Choose is about 15%. That means, 75 out of every 100 projects are not fully funded and deemed not exciting. The organization's mission is to help teachers fund projects they need, and if 75% of those projects end up not fully funded, is Donor's Choose achieving its mission? The chart above, showing the historical distribution of exciting projects versus nonexciting projects, visually illustrates the problem the organization faces. 

There is a need for management of Donor's Choose to understand what attributes make certain projects exciting and what attributes make donors repeat donors. 
<br></br>
To help the organization, I attempt to build a model to predict whether a new project proposal is exciting or not exciting. Lastly, I used descriptive analysis and clustering to understand what characteristics make certain donors repeat donors.

<h2>Languages and Libraries Used</h2>

- Python
- [List of Libraries](https://github.com/graphshade/donorschoose/blob/main/requirements.txt)

<h2>Environment Used </h2>

- <b>Ubuntu 22.04.1 LTS</b>


<h2>Key Findings:</h2>

1. Exciting projects have at least 25% of great comments on the projects page
2. Exciting projects have least one non-teacher referred donor, aand at least one teacher-referred donor
3. The estimated cost of the project including tip has marginal effect on amount donated


<h2>Recommendations:</h2>

1. First, Donors Choose should rank project proposals on its “find a class to support” page by the magnitude of the predicted probability of excitement of the project. In effect,  projects with a higher probability of excitement get seen more and get more donations per impression
2. Second, Donors Choose should apply the findings around features that influence exciting projects to create a guideline for teachers on how to craft winning project proposals. 

The estimated historical gain from implementing the model is presented in the chart below.

<img src="https://github.com/graphshade/donorschoose/blob/main/results/Historical%20gain%20in%20exciting%20projects%20if%20analytical%20solution%20is%20operationalized.png" />



<h2>Reproducing the Analysis:</h2>

<p align="left">
 
1. Clone the project: Run this from the command line
 
 ```commandline
 git clone https://github.com/graphshade/donorschoose.git
 ```
 
2. Set up a virtual environment and install the [required libraries](https://github.com/graphshade/donorschoose/blob/main/requirements.txt)

3. Run the donors_choose.ipynb in the script folder using Jupyter notebook
 
4. Because the dataset is large, use the tuning_all_params.py file in the param_tuning_job folder to tune the models via a batch job.
