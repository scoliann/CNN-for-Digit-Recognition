This script creates a Convolutional Neural Network using TensorFlow for the purpose of digit recognition on the popular MNIST data set.  

## Inspiration
This script was inspired by my desire and progress towards learning TensorFlow.  The code here was heavily borrowed from the provided code and exercises in the [TensorFlow and deep earning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) tutorial on Codelabs.

## Modifications
Many MNIST notebooks and scripts on Kaggle import the data directly from a pre-built TensorFlow module.  This method is very fraudulent for two reasons:

1.  The TensorFlow imports are such that the set of training data is much larger than that provided for the competition on Kaggle.
2.  Importing your data with a pre-built module will do nothing to help you learn how to import and format your data for any other machine learning problem.

## Submission
My CNN got 98.357% of the digit classifications correct.  I am happy with this result, but was hoping for a figure in the 99% range, as this is what was obtained doing the exercises on Codelabs.  One factor that might account for the difference is the fact that on Codelabs, we were using the pre-built TensorFlow imports for MNIST, which, as described before, have more training data.

## The Future
It was cool to learn how to make a CNN from scratch.  I am not sure how necessary this will be in the future, as TensorFlow provides a powerful tool called Inception for image classification tasks.  Inception (as I understand it) is an awesome CNN that Google built training on tons of data.  The final bit of this CNN can be retrained for classification tasks of your choosing.  This both seems easier and seems like it would be a more powerful tool for image classification in the future.  I will be attempting to use Inception to make a classifier to distinguish between men and women, and may go back and try to apply Inception to the Kaggle MNIST competition to see how its performance compares.
