import cv2
import mediapipe as mp
import numpy as np
import math as math
import matplotlib.pyplot as plt
import os
from skimage.filters import unsharp_mask


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__   =  mode
        self.__maxHands__   =  maxHands
        self.__detectionCon__   =   detectionCon
        self.__trackCon__   =   trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
        # duim tot 4 - wijsvinger tot 8 - middel tot 12 - -ring tot 16 - pink tot 20
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList =[]
        yList =[]
        bbox = []
        self.lmsList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
            
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                               (0, 255 , 0) , 2)
        return self.lmsList, bbox
    
    def findFingerUp(self):
         fingers=[]

         if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
              fingers.append(1)
         else:
              fingers.append(0)

         for id in range(1, 5):            
              if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
                   fingers.append(1)
              else:
                   fingers.append(0)
        
         return fingers

    def findDistance(self, p1, p2, frame, draw= True, r=15, t=3):
         
        x1 , y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx , cy = (x1+x2)//2 , (y1 + y2)//2

        if draw:
              cv2.line(frame,(x1, y1),(x2,y2) ,(255,0,255), t)
              cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
              cv2.circle(frame,(x2,y2),r, (255,0,0),cv2.FILLED)
              cv2.circle(frame,(cx,cy), r,(0,0.255),cv2.FILLED)
        len= math.hypot(x2-x1,y2-y1)

        return len, frame , [x1, y1, x2, y2, cx, cy]
# Functie om de grootte van een afbeelding in te stellen
def set_image_size_to_reference(cropped_img, reference_image_path):
    reference_img = cv2.imread(reference_image_path)
    if reference_img is None:
        print(f"Kan referentie-afbeelding niet laden: {reference_image_path}")
        return cropped_img
    # Haal de afmetingen van de referentie-afbeelding op
    reference_height, reference_width = reference_img.shape[:2]
    # Schaal de afbeelding naar de referentie-afmetingen
    resized_img = cv2.resize(cropped_img, (reference_width, reference_height))
    return resized_img

def main():
    image_path = "C:\hogent\Stage\CV\ik7.jpg" 
    frame = cv2.imread(image_path)

    if frame is None:
        print("Afbeelding kan niet worden geladen.")
        return
    
    detector = HandTrackingDynamic()

    # schaal de afbeelding naar een kleinere grootte voor weergave
    height, width = frame.shape[:2]
    new_width = 640 
    new_height = int((new_width / width) * height)  

    frame_resized = cv2.resize(frame, (new_width, new_height))

    
    frame_resized = detector.findFingers(frame_resized)
    lmsList, bbox = detector.findPosition(frame_resized)
    
    # Toon de resultaten
    if len(lmsList) != 0:
        fingers = detector.findFingerUp()
        print(f"Vingers omhoog: {fingers.count(1)}")

    print(f"De lmsList: {lmsList}")

    cv2.putText(frame_resized, str(int(len(lmsList))), (10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    # cv2.imshow("afb", frame_resized)

    """enkel vingertoppen tonen"""
    fingertip_ids = [4, 8, 12, 16, 20]

    cropped_images = []

    reference_image_path = "C:\\hogent\\Stage\\CV\\fingerprint_db2\\vingertop_ID_4_0.jpg"  # Het pad naar de referentieafbeelding

    for id in fingertip_ids:
        cx, cy = lmsList[id][1], lmsList[id][2]

        # Coördinaten ophalen voor uitsnijding
        size = 50  # Pas dit aan op basis van de resolutie
        xmin, ymin = max(0, cx - size), max(0, cy - size)
        xmax, ymax = min(frame_resized.shape[1], cx + size), min(frame_resized.shape[0], cy + size)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        cropped_img = frame_resized[ymin:ymax, xmin:xmax]
        # cv2.imshow("afb",cropped_img )
        # Pas unsharp mask toe voor scherpte
        cropped_img_sharpened = unsharp_mask(cropped_img, radius=6, amount=3)
        cropped_img_sharpened = np.clip(cropped_img_sharpened * 255, 0, 255).astype(np.uint8)
        # Schaal de afbeelding naar dezelfde grootte als de referentie-afbeelding
        cropped_img_resized = set_image_size_to_reference(cropped_img_sharpened, reference_image_path)

        # Create the sharpening kernel 
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
        
        # Sharpen the image 
        cropped_img_resized = cv2.filter2D(cropped_img_resized, -1, kernel) 
        # cv2.imshow("afb", cropped_img_resized)
        # Opslaan in lijst voor visualisatie
        if cropped_img.size > 0:
            cropped_images.append((cropped_img_resized, id))
    
    print(f"lengte cropped: {len(cropped_images)}")



    # visualiseer alle vingertopbeelden naast elkaar
    fig, axes = plt.subplots(1, len(cropped_images), figsize=(15, 5))
    print (f"lengte van alle aparte afbeeldingen tesamen: {len(cropped_images)}")
    for ax, (img, finger_id) in zip(axes, cropped_images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Vingertop ID {finger_id}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # vingertop afbeelding opslaan: 
    output_folder = "vingertop_images"
    os.makedirs(output_folder, exist_ok=True)

    for idx, (img, finger_id) in enumerate(cropped_images):
        filename = f"{output_folder}/vingertop_ID_{finger_id}_{idx}.png"
        cv2.imwrite(filename, img)
        print(f"Afbeelding opgeslagen als: {filename}")

if __name__ == "__main__":
    main()
