# Operationalizing Machine Learning on SageMaker

## Step 1: Training and deployment on Sagemaker

### Initial Setup & Sagemaker Dashboard

I choosed the `ml.t2.medium` instance because it is the cheapest instance and the notebook will not do heavy tasks but rather start other instances to take over tasks like model training and hosting of end points for inference. This instance is also free for starter accounts (250h).

![](images/step_1_notebook_instance.png)

### Download data to an S3 bucket

After executing the first three cells of the notebook, the data is uploaded to the S3 bucket:

![](images/step_1_s3_bucket.png)

### Training and Deployment

These are the training jobs for hyperparameter tuning:

![](images/step_1_hyper_tune.png)

Then, a training job with the best parameters were created:

![](images/step_1_best_estimator_single.png)

As soon as all cells are executed, an endpoint is deployed:

![](images/step_1_end_point_single_instance.png)

And the endpoint for the model trained on multiple instances:

![](images/step_1_end_point_single.png)

After that, a training job with 4 instances were created:

![](images/step_1_best_estimator_multi.png)

And the corresponding end point:

![](images/step_1_end_point_multi.png)

## Step 2: EC2 Training




