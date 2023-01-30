import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


single = cv.imread("face.png", 0)
# many = cv.imread("many_cereals.jpg", 0)

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    sift = cv.SIFT_create()

    key_points1, descriptors1 = sift.detectAndCompute(single, None)
    key_points2, descriptors2 = sift.detectAndCompute(frame, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    best = []

    for m1, m2, in matches:
        if m1.distance < 0.8 * m2.distance:
            best.append([m1])

    print(f"all: {len(matches)}, best: {len(best)}")

    if len(best) > 5:
        src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)
        M, hmask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        h, w = single.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        result = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        cv.imshow('res', result)
        k = cv.waitKey(10)
        if k > 0:
            if chr(k) == 'q':
                break
    else:
        print("Not enough matches")
        mask = None

    matches_images = cv.drawMatchesKnn(single, key_points1, frame, key_points2, best, None)

    # matches_images = cv.drawMatches(single, key_points1, many, key_points2, best, None)

    cv.imshow('frame', frame)
    k = cv.waitKey(10)
    if k > 0:
        if chr(k) == 'q':
            break
