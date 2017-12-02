# TEAM PROJECT FOR CS279

## Automated Classification of Pigmented Lesions 

We will perform automated classification on dermoscopy images obtained from phase 3 of [ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a). 

## Approach

We will represent each image as a combination of textural features, color information encoded in a lower dimensional space, and shape features. A support vector machine will be used to classify the dermoscopy images as melanoma or benign nevus. 10-fold CV will be performed over our initial training set of 2002 dermoscopy images, and final validation will be performed using a hold-out test set. 

## Performance 


### To-do: 
* Extract texture features *(WIP)*
* Convert LBP into scale adaptive *(WIP)*
* Determine how to pre-process hair, foreign bodies *(WIP)*
* Extract color features *(WIP)*	
* Do a classify *(WIP)*
=======

