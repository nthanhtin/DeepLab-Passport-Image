# Passport Image Generator with Harr Cascade and DeepLab

Recently, I have to submit some very important paperwork that requires passport photo along with it. Despite the on going pandemic, I still had to go to Walgreens because I don't have any white screen and I don't know photoshop. So I got there, waited in line for about an hour just to get my picture taken, heard someone coughing, I left instantly and thought to myself: I think I can do this with Machine Learning. I came back 3 hours later straight to the printer with my own pictures costing 23 cents each and out of the store after 5 minutes. Here's how I solved my first word problem and protect my health with Machine Learning.

# Usage:

`python passport_photo.py -i input_image -o output_image'

# Face Detection

I needed a quick and painless way to detect human face so I choose to use Haar feature-based cascade classifiers which available in OpenCV and pretrained xml model can be found [online](https://github.com/opencv/opencv/tree/master/data/haarcascades) too. You can read more about the theory [here](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).

`face_cascade = cv2.CascadeClassifier(args.harr_weight_file)`

Basically, the classifier was trained with with a lot of positive and negative image. The features of the train set then learned and these features can be used to detect the object in new images. The method was introduced in 2001 by Paul Viola and Michael Jones, despite being around for a while, it still holding up very well in practical use.

`faces = face_cascade.detectMultiScale(gray, 1.3, 4)`

While testing the code, I found that having scaleFactor at 1.30 and minNeighbors at 4 works well and fast for me since faces in potraits are usually very big. you can decrease them if the code doesn't work so well for you but the code will be slower.

Result:

![](/asset/s1.png)

## Center and Crop
```
if w>=h:
    width_crop = (height-(w_f))/2
    if width_crop.is_integer():
        img = img[:, x_f-int(width_crop):x_f+w_f+int(width_crop)]
    else:
        img = img[:, x_f-int(np.floor(width_crop)):x_f+w_f+int(np.ceil(width_crop))]
else:
    height_crop = (width-(h_f))/2
    if height_crop.is_integer():
        img = img[y_f-int(height_crop):y_f+h_f+int(height_crop), :]
    else:
        img = img[y_f-int(np.floor(height_crop)):y_f+h_f+int(np.ceil(height_crop)), :] 
```
Passport photo has to be 2x2 right. This is very simple. I just find the difference between the face bounding box returned by the classifier and crop my image accordingly. However, I know there will be some cases where this will go out of bound. Feel free to comment or open an issue on Github with sugession. I know this can be improved since it's heavily depends on if your face is perpendicular to the camera or not. 
Result:

![](/asset/s2.png)
## DeepLab for White Background

In detail on [Google Blog](https://github.com/tensorflow/models/tree/master/research/deeplab)
I used DeepLab v3 for the task. It was trained on the COCO dataset with a great peformance in image segmentation task.

![](/asset/deeplab.png)

For my best result, I used the pretrained Xception weights but Mobilenetv2 is way faster with a bit of accuracy trade off. 

![](/asset/seg.png)

After having segmentation matrix the rest is imple, I used it as a mask and multiply the background with `1.0-mask` to get the white background, multiply `mask` with original image to get the object and add them together to get the final image:

![](/asset/final.png)
