import cv2

# matching with db
fingerprint_db_img = cv2.imread("./skeleton.png")
fingerprint_live = cv2.imread("./skeleton2.png")

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_live, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_db_img, None)

print(f"Keypoints in fingerprint_img: {len(keypoints_1)}")
print(f"Keypoints in fingerprint_db_img: {len(keypoints_2)}")


if descriptors_1 is None or descriptors_2 is None:
    print("Error: No descriptors found in the image")
    exit(1)

# finding nearest match with KNN algorithm 

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict())
matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
match_points = []
   
for p, q in matches:
    if p.distance < 0.75*q.distance:
        match_points.append(p)

# keypoint detection / detecting the fingerprint matched ID
keypoints = 0
if len(keypoints_1) <= len(keypoints_2):
    keypoints = len(keypoints_1)            
else:
    keypoints = len(keypoints_2)

# if (len(match_points) / keypoints)>0.95:
print("% match: ", len(match_points) / keypoints * 100)

# geen idee waarom 
print("Fingerprint ID: " + str(keypoints)) 

result = cv2.drawMatchesKnn(fingerprint_live, keypoints_1, fingerprint_db_img, 
                            keypoints_2, matches1to2=[match_points], outImg=None, matchColor=(0, 155, 0), 
                             singlePointColor=(0, 255, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
result = cv2.resize(result, None, fx=1, fy=1)

# de twee afbeeldingen bij elkaar zetten en zo vergelijken zodat je een overzicht hebt 

cv2.namedWindow("result.jpg", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("result.jpg", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("result.jpg", result)
cv2.waitKey()
cv2.destroyAllWindows()
