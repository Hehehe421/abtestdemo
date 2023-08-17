# ABtestdemo
## Overview
During today's demonstration, we will delve into the intricacies of constructing a machine learning (ML) model and subsequently conducting AB testing using Databricks. To provide a contextual framework for our demo, let us consider the scenario of Airbnb:

Demo Use Case: Within the realm of improving the host experience on their website, Airbnb aims to enhance the process of listing creation. To achieve this, they have conceived the idea of constructing a predictive model that can accurately estimate the most suitable listing price based on various factors such as location, number of rooms, and other relevant features. This predictive model will serve as a reference point for hosts when they create new listings, ultimately streamlining the process and reducing time consumption.

To further optimize their efforts, Airbnb has developed two distinct versions of the ML model, each employing different techniques to predict the listing price. In order to determine which version yields superior results and is more conducive to their business objectives, Airbnb intends to conduct real-time AB testing, thereby comparing the efficacy of the two models in a practical setting with actual customers.

The primary metrics that Airbnb seeks to improve through AB testing are the website's activity rate and the total duration engaged on the website. By gauging the impact of the ML models on these metrics, Airbnb can ascertain which version better aligns with their overarching goals, ultimately fostering a more engaging and efficient platform for their users.

## What is AB testing?
An A/B test, also called a controlled experiment or a randomized control trial, is a statistical method of determining which of a set of variants is the best. A/B tests allow organizations and policy-makers to make smarter, data-driven decisions that are less dependent on guesswork.

Today, A/B tests are an important business tool, used to make decisions in areas like product pricing, website design, marketing campaign design, and brand messaging. A/B testing lets organizations quickly experiment and iterate in order to continually improve their business.

In data science, A/B tests can also be used to choose between two models in production, by measuring which model performs better in the real world. In this formulation, the control is often an existing model that is currently in production. The treatment is a new model being considered to replace the old one. We would like to using A/B test to test the effectiveness of these two model on the organization OEC (overall evaulation criteria).

## Medium Blogs
- [Building ML Predictive Models and Managing AB Testing with Databricks - Part I](https://levelup.gitconnected.com/building-ml-predictive-models-and-managing-ab-testing-with-databricks-76460834d6a3)
- [Building ML Predictive Models and Managing AB Testing with Databricks - Part II](https://medium.com/p/54ab87816d4d)
