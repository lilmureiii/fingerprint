import cv2
import os

# matching with db
fingerprint_live = cv2.imread("./fingerprint_v2/skeleton3.png")
db_dir = "./fingerprint_v2"

sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_live, None)

print(f"Keypoints in fingerprint_img: {len(keypoints_1)}")


if descriptors_1 is None:
    print("Error: No descriptors found in the image")
    exit(1)

# finding nearest match with KNN algorithm 

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict())
match_res = []

for filename in os.listdir(db_dir):
    filepath = os.path.join(db_dir, filename)
    fingerprint_db_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if fingerprint_db_img is None:
        print(f"Warning: Could not read {filename}")
        continue

    # SIFT detectie en descriptors
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_db_img, None)

    if descriptors_2 is None:
        print(f"Warning: No descriptors found in {filename}")
        continue

    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = [p for p, q in matches if p.distance < 0.75 * q.distance]

    # keypoint detection / detecting the fingerprint matched ID
    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)            
    else:
        keypoints = len(keypoints_2)

    match_perc = len(match_points) / keypoints * 100
    match_res.append((filename, match_perc, fingerprint_db_img, keypoints_2, match_points))

# if (len(match_points) / keypoints)>0.95:
print("% match: ", len(match_points) / keypoints * 100)

# geen idee waarom 
print("Fingerprint ID: " + str(keypoints)) 


match_res.sort(key=lambda x: x[1], reverse=True)

# Top 3 matches selecteren
top_matches = match_res[:3]

for i, (filename, match_percentage, db_img, keypoints_2, match_points) in enumerate(top_matches):
    print(f"Match {i+1}: {filename} - {match_percentage:.2f}%")
    result = cv2.drawMatchesKnn(fingerprint_live, keypoints_1, db_img, 
                                keypoints_2, matches1to2=[match_points], outImg=None, matchColor=(0, 255, 0), 
                                singlePointColor=(0, 255, 255),flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # result = cv2.resize(result, None, fx=1, fy=1)

# de twee afbeeldingen bij elkaar zetten en zo vergelijken zodat je een overzicht hebt 

    # cv2.namedWindow("result.jpg", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("result.jpg", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(f"Match {i+1}: {filename}", result)

cv2.waitKey()
cv2.destroyAllWindows()
