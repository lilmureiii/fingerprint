
def  blub():# ------------------------- fingerprint recognition ----------------------------
    path1 = "./fingerprint_db2/thumb.jpg"
    dir = "./fingerprint_db2"

    # # preprocessor-object aan en verwerk de afbeelding
    # fingerprint = FingerprintPreprocessor(path)
    # thinned, skeleton = fingerprint.preprocess()


    def should_skip_by_ratio(minutiae1, minutiae2, ratio_threshold=1.5):
        """
        Return True if the ratio of the Euclidean distances between two minutiae points 
        exceeds the ratio_threshold.
        """
        for m1 in minutiae1:
            for m2 in minutiae2:
                distance1 = np.linalg.norm(np.array(m1) - np.array(m2))
                for m3 in minutiae1:
                    for m4 in minutiae2:
                        distance2 = np.linalg.norm(np.array(m3) - np.array(m4))
                        if distance1 != 0 and distance2 != 0:
                            ratio = max(distance1, distance2) / \
                                min(distance1, distance2)
                            if ratio > ratio_threshold:
                                return True
        return False


    def match_minutiae(minutiae1, minutiae2, threshold=10):
        """
        compare minutiae lists of 2 fingerprints by using the euclidean distance: 
            np.linalg.norm(np.array(m1) - np.array(m2)) < threshold

        use a set so that it doesn't match the same points several times
        """
        matches = []
        matched_points_1 = set()
        matched_points_2 = set()

        for i, m1 in enumerate(minutiae1):
            for j, m2 in enumerate(minutiae2):
                # Bereken de Euclidische afstand tussen de coördinaten (x, y) van m1 en m2
                # Gebruik alleen de x, y coördinaten
                distance = np.linalg.norm(np.array(m1[:2]) - np.array(m2[:2]))
                if i not in matched_points_1 and j not in matched_points_2 and distance < threshold:
                    matches.append((m1, m2))
                    matched_points_1.add(i)
                    matched_points_2.add(j)

        return len(matches) / max(len(minutiae1), len(minutiae2)) * 100


    def extract_minutiae(image_path):
        """extract minutiae points from a fingerprint image."""
        processor = FingerprintPreprocessor(image_path)
        processor.preprocess()
        processor.calculate_minutiaes()
        return processor.minutiaes, processor.thinned




        # Visualiseer alle thinned afbeeldingen
        # fig, axes = plt.subplots(1, len(thinned_images), figsize=(15, 5))
        # for ax, img, filename in zip(axes, thinned_images, filenames):
        #     ax.imshow(img, cmap='gray')
        #     ax.set_title(f"Thinned: {filename}")
        #     ax.axis('off')

        # plt.tight_layout()
        # plt.show()

        # print("Alle thinned afbeeldingen zijn verwerkt en weergegeven!")

    # def match_fingerprint(self):
    #     minutiae1, thinned1 = extract_minutiae(path1)
    #     matches = []
    #     for file in os.listdir(dir):
    #         if file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #             file_path = os.path.join(dir, file)
    #             minutiae2, thinned2 = extract_minutiae(file_path)
    #             match_score = match_minutiae(minutiae1, minutiae2)
    #             matches.append((file, match_score))

    #         # print(f"Minutiae match percentage: {match_score:.2f}%")

    #     print("Alle matches: ", matches)
    #     highest = None
    #     matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)
    #     print("Alle matches gesorteerd: ", matches_sorted)

    #     # Haal de hoogste match op
    #     threshold = 95
    #     highest = matches_sorted[0]
    #     if highest[1] >= threshold:
    #         print("Best match is", highest)
    #     else:
    #         print("This person has no fingerprints in the database and therefore can't be identified in the system")
