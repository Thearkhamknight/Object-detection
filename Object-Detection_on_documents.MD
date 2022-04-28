If you would like to test the performance of this Object-detection notebook on document data, I provided it as a binary asset with the release of this
repository with this link: https://github.com/Thearkhamknight/Object-detection/releases/tag/v0.0-alpha . The name of that notebook is 
TF_exec_doc_inference.nbconvert.ipynb. There are a few changes that I made to the original version. The annotations and images subdirectories corresponding 
to the document data inside my cloned driving-object-detection repository are titled Annotations and Images 
(instead of annotations and images for the traffic light data). I was able to provide all the test set images inside images, 
however I was not able to provide the images and training set because of file size upload constraints on neither the Github repository nor as a binary asset in the release (both the training set and the entire image set are well over 1000 images). 

However, I will provide a link for both the entire set of images and the training set as zip files that you can access through Google Drive.
Here is the link for both of the zip files : https://drive.google.com/drive/folders/1YSoObzOqhNoJYP6NuV2Cy82QzEHXVZH-?usp=sharing .
I will also specify the few changes I made to the original version. The cell numbers I use here correspond to TF_exec_doc_inference.nbconvert.ipynb.
I added two new cells after cell 18. The first cell, cell 19 removes the images and annotations subdirectories inside driving-object-detection because you will not need them. Cell 20 extracts the images and train zip files to driving-object-detection/Images/ and
driving-object-detection/Images/train/ respectively. Absolute file paths are necessary here so you will need to specify them, all you have to do is change 
the preceding directories before Object-detection. Finally, cells 22 and 23, the two cells after printing the pipeline.config document are commented out. 
This is because the document data is already partitioned into a training and a test set and the data is already in csv format. Keep these cells 
commented. Everything else can be kept the same and should run smoothly.
