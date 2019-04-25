import numpy as np
import cv2
import time
cap = cv2.VideoCapture('hty1.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.mp4',fourcc, 20.0, (640,480))

template1= cv2.imread('cam_dung_va_do_xe_0.png')
template1= cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
template2= cv2.imread('cam_dung_va_do_xe_1.png')
template2= cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
template3= cv2.imread('cam_dung_va_do_xe_2.png')
template3= cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
template4= cv2.imread('cam_dung_va_do_xe_3.png')
template4= cv2.cvtColor(template4, cv2.COLOR_BGR2GRAY)
# template5= cv2.imread('2cammoto.png')
# template5= cv2.cvtColor(template5, cv2.COLOR_BGR2GRAY)
# template6= cv2.imread('2dithang4.png')
# template6= cv2.cvtColor(template6, cv2.COLOR_BGR2GRAY)
# template7= cv2.imread('2dithang1.png')
# template7= cv2.cvtColor(template7, cv2.COLOR_BGR2GRAY)
# template8= cv2.imread('2dithang2.png')
# template8= cv2.cvtColor(template8, cv2.COLOR_BGR2GRAY)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

frame_rate = 30
prev = 0

while(cap.isOpened()):
    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()
    
    w1,h1 =template1.shape[::-1]
    w2,h2 =template2.shape[::-1]
    w3,h3 =template3.shape[::-1]
    w4,h4 =template4.shape[::-1]
    # w5,h5 =template5.shape[::-1]
    # w6,h6 =template6.shape[::-1]
    # w7,h7 =template5.shape[::-1]
    # w8,h8 =template6.shape[::-1]
    
    ret, frame = cap.read()
    
    frame = rescale_frame(frame, percent=35)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    threshold=0.8 
    res1 = cv2.matchTemplate(frame_gray, template1, cv2.TM_CCOEFF_NORMED)
    loc1 = np.where(res1 >= threshold)
    
    res2 = cv2.matchTemplate(frame_gray, template2, cv2.TM_CCOEFF_NORMED)
    loc2 = np.where(res2 >= threshold)

    res3 = cv2.matchTemplate(frame_gray, template3, cv2.TM_CCOEFF_NORMED)
    loc3 = np.where(res3 >= threshold)

    res4 = cv2.matchTemplate(frame_gray, template4, cv2.TM_CCOEFF_NORMED)
    loc4 = np.where(res4 >= threshold)

    # res5 = cv2.matchTemplate(frame_gray, template5, cv2.TM_CCOEFF_NORMED)
    # loc5 = np.where(res5 >= threshold)

    # res6 = cv2.matchTemplate(frame_gray, template6, cv2.TM_CCOEFF_NORMED)
    # loc6 = np.where(res6 >= threshold)

    # res7 = cv2.matchTemplate(frame_gray, template7, cv2.TM_CCOEFF_NORMED)
    # loc7 = np.where(res7 >= threshold)

    # res8 = cv2.matchTemplate(frame_gray, template8, cv2.TM_CCOEFF_NORMED)
    # loc8 = np.where(res8 >= threshold)

    
    for pt1 in zip(*loc1[::-1]):
        
        cv2.rectangle(frame, pt1, (pt1[0]+w1, pt1[1]+h1),(0,255,0),2)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2
        cv2.putText(frame,'VONG SANG TRAI', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

    for pt2 in zip(*loc2[::-1]):
        cv2.rectangle(frame, pt2, (pt2[0]+w2, pt2[1]+h2),(0,255,0),2)       
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2 
        cv2.putText(frame,'DI THANG', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    for pt3 in zip(*loc3[::-1]):
        cv2.rectangle(frame, pt3, (pt3[0]+w3, pt3[1]+h3),(0,255,0),2)        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2
        cv2.putText(frame,'CAM DUNG',bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

    for pt4 in zip(*loc4[::-1]):
        cv2.rectangle(frame, pt4, (pt4[0]+w4, pt4[1]+h4),(0,255,0),2)       
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (100,100)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 2
        cv2.putText(frame,'CAM DUNG', bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    # for pt5 in zip(*loc5[::-1]):
    #     cv2.rectangle(frame, pt5, (pt5[0]+w5, pt5[1]+h5),(0,255,0),2)
        
    #     font                   = cv2.FONT_HERSHEY_SIMPLEX
    #     bottomLeftCornerOfText = (100,100)
    #     fontScale              = 1
    #     fontColor              = (0,255,0)
    #     lineType               = 2

    #     cv2.putText(frame,'CAM MOTO', 
    #     bottomLeftCornerOfText, 
    #     font, 
    #     fontScale,
    #     fontColor,
    #     lineType)
    # for pt6 in zip(*loc6[::-1]):
    #     cv2.rectangle(frame, pt6, (pt6[0]+w6, pt6[1]+h6),(0,255,0),2)
        
    #     font                   = cv2.FONT_HERSHEY_SIMPLEX
    #     bottomLeftCornerOfText = (100,100)
    #     fontScale              = 1
    #     fontColor              = (0,255,0)
    #     lineType               = 2

    #     cv2.putText(frame,'DI THANG', 
    #     bottomLeftCornerOfText, 
    #     font, 
    #     fontScale,
    #     fontColor,
    #     lineType)
    # for pt7 in zip(*loc7[::-1]):
    #     cv2.rectangle(frame, pt7, (pt7[0]+w7, pt7[1]+h7),(0,255,0),2)
        
    #     font                   = cv2.FONT_HERSHEY_SIMPLEX
    #     bottomLeftCornerOfText = (100,100)
    #     fontScale              = 1
    #     fontColor              = (0,255,0)
    #     lineType               = 2

    #     cv2.putText(frame,'DI THANG', 
    #     bottomLeftCornerOfText, 
    #     font, 
    #     fontScale,
    #     fontColor,
    #     lineType)
    # for pt8 in zip(*loc8[::-1]):
    #     cv2.rectangle(frame, pt8, (pt8[0]+w8, pt8[1]+h8),(0,255,0),2)
        
    #     font                   = cv2.FONT_HERSHEY_SIMPLEX
    #     bottomLeftCornerOfText = (100,100)
    #     fontScale              = 1
    #     fontColor              = (0,255,0)
    #     lineType               = 2

    #     cv2.putText(frame,'DI THANG', 
    #     bottomLeftCornerOfText, 
    #     font, 
    #     fontScale,
    #     fontColor,
    #     lineType)

    if ret==True:                
        # write the  frame
        out.write(frame)        
        cv2.imshow('frame',frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
