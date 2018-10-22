import cv2 as cv
import numpy as np

image_variable=cv.imread("digits.png")
image_grey=cv.cvtColor(image_variable,cv.COLOR_BGR2GRAY)

image_cells=[np.hsplit(row,100) for row in np.split(image_grey,50)]

array_variable=np.array(image_cells)
print("The shape of our cell array"+str(array_variable.shape))

data_train=array_variable[:,:70].reshape(-1,400).astype(np.float32)
data_test=array_variable[:,70:100].reshape(-1,400).astype(np.float32)

dataset_variable=[0,1,2,3,4,5,6,7,8,9]
data_train_label=np.repeat(dataset_variable,350)[:,np.newaxis]
data_test_label=np.repeat(dataset_variable,150)[:,np.newaxis]

knn=cv.ml.KNearest_create()
knn.train(data_train,cv.ml.ROW_SAMPLE,data_train_label)

return_variable,result_variable,neighbours_variable,distance_variable=knn.findNearest(data_test,k=3)

matches_variable=result_variable==data_test_label
correct_matches_variable=np.count_nonzero(matches_variable)
accuracy_correct_matches_variable=correct_matches_variable*(100.0/result_variable.size)
print("Accuracy is %.2f"% + accuracy_correct_matches_variable + "%")

def array_contour(data_contour):
    if cv.contourArea(data_contour)>10:
        moments_variable=cv.moments(data_contour)
        return int(moments_variable['m10']/moments_variable['m00'])
    else:
        return int(0)

def square_check(not_square):
    set_variable=[0,0,0]
    image_dimension=not_square.shape
    x_dimension_variable=image_dimension[0]
    y_dimension_variable=image_dimension[1]
    if(x_dimension_variable == y_dimension_variable):
        square_variable=not_square
        return square_variable
    else:
        double_size_variable=cv.resize(not_square,(2*y_dimension_variable,2*x_dimension_variable),interpolation=cv.INTER_CUBIC)
        x_dimension_variable=x_dimension_variable*2
        y_dimension_variable=y_dimension_variable*2

        if(x_dimension_variable > y_dimension_variable):
            pad_variable=int((x_dimension_variable-y_dimension_variable)/2)
            double_size_square_variable=cv.copyMakeBorder(double_size_variable,0,0,pad_variable,pad_variable,cv.BORDER_CONSTANT,value=set_variable)
            cv.copyMakeBorder(double_size_square_variable,0,0,0,0,0)
        else:
            pad_variable=int((y_dimension_variable-x_dimension_variable)/2)
            double_size_square_variable=cv.copyMakeBorder(double_size_variable,0,0,pad_variable,pad_variable,cv.BORDER_CONSTANT,value=set_variable)

    return double_size_square_variable

def resize_pixel(dimensions_image,image_parameter):
    buffer_pixels_variable=4
    dimensions_image=dimensions_image-buffer_pixels_variable
    squared_image=image_parameter
    r_float_variable=float(dimensions_image)/squared_image.shape[1]
    dimension_dim=(dimensions_image,int(squared_image.shape[0]*r_float_variable))
    resized_image=cv.resize(image_parameter,dimension_dim,interpolation=cv.INTER_AREA)
    image_dimension_r=resized_image.shape
    x_dimension_r=image_dimension_r[0]
    y_dimension_r=image_dimension_r[1]
    set_variable_r=[0,0,0]
    if(x_dimension_r>y_dimension_r):
        resized_image=cv.copyMakeBorder(resized_image,0,0,0,1,cv.BORDER_CONSTANT,value=set_variable_r)
    if(x_dimension_r<y_dimension_r):
        resized_image=cv.copyMakeBorder(resized_image,1,0,0,0,cv.BORDER_CONSTANT,value=set_variable_r)
    resized_image_r=cv.copyMakeBorder(resized_image,2,2,2,2,cv.BORDER_CONSTANT,value=set_variable_r)
    image_dimension_2_r=resized_image_r.shape
    x_image_dimension_r=image_dimension_2_r[0]
    y_image_dimension_r=image_dimension_2_r[1]
    return resized_image_r

image_variable_2=cv.imread("num.png")
image_grey_2=cv.cvtColor(image_variable_2,cv.COLOR_BGR2GRAY)
cv.imshow("Image 2",image_variable_2)
cv.imshow("Grey 2", image_grey_2)
cv.waitKey(0)

image_blurred=cv.GaussianBlur(image_grey_2,(5,5),0)
cv.imshow("Blurred Image", image_blurred)
cv.waitKey(0)

image_edged=cv.Canny(image_blurred,0,80)
cv.imshow("Canny Edged", image_edged)
cv.waitKey(0)

_, contours_image, _=cv.findContours(image_edged.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contours_image=sorted(contours_image, key=array_contour, reverse=False)

full_number=[]

for c in contours_image:
    (x,y,w,h)=cv.boundingRect(c)

    cv.drawContours(image_variable_2, contours_image,-1,(0,255,0),1)
    cv.imshow("Countours",image_variable_2)

    if w >= 5 and h >= 25:

        roi = image_blurred[y:y + h, x:x +w]

        ret, roi = cv.threshold(roi,127,255,cv.THRESH_BINARY_INV)
        squared = square_check(roi)
        final = resize_pixel(20, squared)
        cv.imshow("Final Image",final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbors, dist=knn.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        cv.rectangle(image_variable_2,(x,y),(x+w,y+h),(0,0,255),2)
        cv.putText(image_variable_2,number,(x,y+155),
                   cv.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
        cv.imshow("Image2",image_variable_2)
        cv.waitKey(0)

cv.destroyAllWindows()

