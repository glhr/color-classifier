* Load image dataset of colored objects
* Extract features from each image (currently, the feature vector is obtained by down-sampling the image to 10x10x3 and then flattening the array)
* Save features and corresponding color class (eg. orange, blue) into JSON file
* Load training data from JSON file and train SDGClassifier
* Predict color of unseen image
