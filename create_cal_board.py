import cv2 as cv
import numpy as np
import apriltag as at
import os

def random_apriltag(tag_list,tag_location):
    num_of_tags = len(tag_list)
    tag_id = np.random.randint(2, num_of_tags-1)
    tag_filename = tag_list[tag_id]
    tag_image = cv.imread(tag_location + tag_filename, cv.IMREAD_GRAYSCALE)
    assert tag_image is not None, "Failed to load tag image at {}".format(tag_filename)
    tag_number = int(tag_filename.split('_')[-1].split('.')[0])
    return tag_image, tag_number

def create_checkerboard_with_apriltags(rows, cols, square_size, tag_location):
    # Create a blank checkerboard image
    board_size = (rows * square_size, cols * square_size)
    checkerboard = np.zeros(board_size, dtype=np.uint8)
    id_matrix = np.zeros((rows, cols))
    tag_list = os.listdir(tag_location)[2:]
    tag_list.sort()
    # Iterate over each square in the checkerboard
    for row in range(rows):
        for col in range(cols):
            top_left = (row * square_size, col * square_size)
            bottom_right = ((row + 1) * square_size, (col + 1) * square_size)
            
            if (row + col) % 2 != 0:
                # White square - embed an AprilTag
                tag_image, tag_number = random_apriltag(tag_list, tag_location)
                tag_image_resized = cv.resize(tag_image, (square_size, square_size))
                checkerboard[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = tag_image_resized
                id_matrix[row, col] = tag_number
            else:
                # Black square
                checkerboard[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 0
    
    checkerboard = cv.resize(checkerboard, (cols * 100, rows * 100), interpolation=cv.INTER_NEAREST)
    return checkerboard, id_matrix

if __name__ == '__main__':
    # Parameters
    rows = 7  # Number of rows in the checkerboard
    cols = 14  # Number of columns in the checkerboard
    tag_size = 10  # Size of the april tag
    tag_location = "apriltag-imgs/tag36h11/"

    # Generate the checkerboard with embedded AprilTags
    checkerboard_with_tags, id_matrix = create_checkerboard_with_apriltags(rows, cols, tag_size, tag_location)
    print(id_matrix)
    np.save('calibration_board.npy', id_matrix)

    # Save and display the result
    cv.imwrite('calibration_board.png', checkerboard_with_tags)
    cv.imshow('Checkerboard with AprilTags', checkerboard_with_tags)
    cv.waitKey(0)
    cv.destroyAllWindows()

