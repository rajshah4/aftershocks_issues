Dear Editors:

A recent paper you published by DeVries, et al., *Deep learning of aftershock patterns following large Earthquakes*, contains significant methodological errors that undermine its conclusion.  These errors should be highlighted, as data science is still an emerging field that hasn’t yet matured to the rigor of other fields.  Additionally, not correcting the published results will stymie research in the area, as it will not be possible for others to match or improve upon the results.  We have contacted the author and shared with them the problems around data leakage, learning curves, and model choice.  They have not yet responded back.

​	First, the results published in the paper, AUC of 0.849, are inflated because of target leakage.   The approach in the paper used part of an earthquake to train the model, which then was used again to test the model.  This form of target leakage can lead to inflated results in machine learning.  To prevent against this, a technique called group partitioning is used.  This requires ensuring an earthquake appears either in the train portion of the data or the test portion.  This is not an unusual methodological mistake, for example a recent paper by Rajpurkar et. al on chest x-rays made the same mistake, where x-rays for an individual patient could be found in both the train and test set.  These authors later revised their paper to correct this mistake.  

In this paper, several earthquakes, including 1985NAHANN01HART, 1996HYUGAx01YAGI, 1997COLFIO01HERN, 1997KAGOSH01HORI,  2010NORTHE01HAYE were represented in both the train and test part of the dataset.  For example, in 1985 two large magnitude earthquakes occurred near the North Nahanni River in the northeast Cordillera, Northwest Territories, Canada, on 5 October (MS 6.6) and 23 December (MS 6.9).  In this dataset, one of the earthquakes is in the train set and the other in the test set.  To ensure the network wasn’t learning the specifics about the regions, we used group partitioning, this ensures an earthquake’s data only was in test or in train and not in both.  If the model was truly learning to predict aftershocks, such a partitioning should not affect the results.  

We applied group partitioning of earthquakes randomly across 10 different runs with different random seeds for the partitioning.    I am happy to share/post the group partitioning along with the revised datasets.  We found the following results as averaged across the 10 runs (~20% validation):



| **Method**                     | **Mean AUC** |
| ------------------------------ | ------------ |
| Coulomb failure stress-change  | 0.60         |
| Maximum change in shear stress | 0.77         |
| von Mises yield criterion      | 0.77         |
| Random Forest                  | 0.76         |
| Neural Network                 | 0.77         |



In terms of predictive performance, the machine learning methods are not an improvement over traditional techniques of the maximum change in shear stress or the von Mises yield criterion.  To assess the value of the deep learning approach, we also compared the performance to a baseline Random Forest algorithm (basic default parameters - 100 trees) and found only a slight improvement.  

It is crucial that the results in the paper will be corrected.  The published results provide an inaccurate portrayal of the results of machine learning / deep learning to predict aftershocks.  Moreover, other researchers will have trouble sharing or publishing results because they cannot meet these published benchmarks.  It is in the interest of progress and transparency that the AUC performance in the paper will be corrected.

The second problem we noted is not using learning curves.  Andrew Ng has popularized the notion of learning curves as a fundamental tool in error analysis for models.  Using learning curves, one can find that training a model on just a small sample of the dataset is enough to get very good performance.  In this case, when I run the neural network with a batch size of 2,000 and 8 steps for one epoch, I find that 16,000 samples are enough to get a good performance of 0.77 AUC.  This suggests that there is a relatively small signal in the dataset that can be found very quickly by the neural network.  This is an important insight and should be noted.  While we have 6 million rows, you can get the insights from just a small portion of that data.

The third issue is jumping straight to a deep learning model without considering baselines.  Most mainstream machine learning papers will use benchmark algorithms, say logistic regression or random forest when discussing new algorithms or approaches.  This paper did not have that.  However, we found that a simple random forest model was able to achieve similar performance to neural network.  This is an important point when using deep learning approaches.  In this case, really any simple model (e.g. SVM, GAM) will provide comparable results.  The paper gives the misleading impression that only deep learning is capable of learning the aftershocks.

As practicing data scientists, we see these sorts of problems on a regular basis.  As a field, data science is still immature and there isn’t the methodological rigor of other fields.  Addressing these errors will provide the research community with a good learning example of common issues practitioners can run into when using machine learning.  The only reason we can learn from this is that the authors were kind enough to share their code and data.  This sort of sharing benefits everyone in the long run.

At this point, I have not publicly shared or posted any of these concerns.  I have shared them with the author and she did not reply back after two weeks.   I thought it would be best to privately share them with you first.  Please let me know what you think. If we do not hear back from you by November 20th, we will make our results public.



Thank you

Rajiv Shah

University of Illinois at Chicago



Lukas Innig

DataRobot