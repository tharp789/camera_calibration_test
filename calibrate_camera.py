import cv2 as cv
import numpy as np
import apriltag as at

def find_checkerboard_corners(camera_image, grid_size):
    # find chessboard corners
    ret, corners = cv.findChessboardCorners(camera_image, grid_size, None)
    if ret == True:
        print("{} corners found".format(corners.shape[0] * corners.shape[1]))
        return corners
    else:
        print("checkerboard corners not found")
        return None

def fill_boxes_white(image, results):
    for r in results:
        (pt1, pt2, pt3, pt4) = r.corners
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        pt3 = (int(pt3[0]), int(pt3[1]))
        pt4 = (int(pt4[0]), int(pt4[1]))
        cv.fillConvexPoly(image, np.array([pt1, pt2, pt3, pt4]), 255)
    return image

def calibrate_camera(camera_image, id_matrix, grid_size=(13, 6)):
    camera_image = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)
    corner_image = camera_image.copy()

    # load apriltag detector
    tag_image = cv.threshold(camera_image, 150, 255, cv.THRESH_BINARY)[1]

    print("Detecting AprilTags...")
    options = at.DetectorOptions(families="tag36h11")
    detector = at.Detector(options)
    results = detector.detect(tag_image)
    expected_ids = id_matrix.flatten().astype(np.uint32)
    expected_ids = expected_ids[expected_ids != 0]
    expected_ids.sort()
    tag_ids = np.array([r.tag_id for r in results])
    tag_ids.sort()
    #check if the detected april tags matched the expected ones
    assert np.array_equal(tag_ids, expected_ids), "AprilTag IDs do not match"
    
    print("{} total AprilTags detected".format(len(results)))

    # with some pictures, it seemed to be needed to block out the apriltags
    # fill boxes with white if needed for detection
    # for r in results:
    #     (pt1, pt2, pt3, pt4) = r.corners
    #     pt1 = (int(pt1[0]), int(pt1[1]))
    #     pt2 = (int(pt2[0]), int(pt2[1]))
    #     pt3 = (int(pt3[0]), int(pt3[1]))
    #     pt4 = (int(pt4[0]), int(pt4[1]))
    #     cv.fillConvexPoly(corner_image, np.array([pt1, pt2, pt3, pt4]), 255)

    # draw results
    print("Finding checkerboard corners...")
    corners = find_checkerboard_corners(corner_image, grid_size)
    if corners is not None:
        corners2 = cv.cornerSubPix(camera_image, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
        rms_error, mtx, dist, rvecs, tvecs = cv.calibrateCamera([obj_points], [corners2], camera_image.shape[::-1], None, None)
        print("Camera calibrated")
        #calcuate reprojection error
        mean_error = 0
        for i in range(len(obj_points)):
            projected_point, _ = cv.projectPoints(obj_points[i], rvecs[0], tvecs[0], mtx, dist)
            projected_point = projected_point[0]
            actual_point = corners2[i]
            error = cv.norm(actual_point, projected_point, cv.NORM_L2)
            mean_error += error
        mean_error /= len(obj_points)
        print("Mean error: ", mean_error)
        print("RMS error: ", rms_error)
    camera_image = cv.cvtColor(camera_image, cv.COLOR_GRAY2BGR)
    if corners is not None:
        camera_image = cv.drawChessboardCorners(camera_image, grid_size, corners, True)
    for r in results:
        (pt1, pt2, pt3, pt4) = r.corners
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        pt3 = (int(pt3[0]), int(pt3[1]))
        pt4 = (int(pt4[0]), int(pt4[1]))
        cv.line(camera_image, pt1, pt2, (0, 255, 0), 2)
        cv.line(camera_image, pt2, pt3, (0, 255, 0), 2)
        cv.line(camera_image, pt3, pt4, (0, 255, 0), 2)
        cv.line(camera_image, pt4, pt1, (0, 255, 0), 2)
        cv.putText(camera_image, str(r.tag_id), (pt1[0], pt1[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    
    cv.imshow('camera_image', camera_image)


if __name__ == '__main__':
    # read camera image
    camera_image = cv.imread('photos/straight_small.jpg')
    assert camera_image is not None, "Failed to load camera image"
    # camera_image = cv.imread('test_board.png')
    id_matrix = np.load('calibration_board_ids.npy')
    # print(id_matrix)
    calibrate_camera(camera_image, id_matrix)
    # wait for key press
    cv.waitKey(0)
    cv.destroyAllWindows()