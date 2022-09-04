# import the needed libraries
import face_recognition
import cv2
import os
import numpy as np

# recognize faces in a video recorded video
video_file = "videos/video.mp4"
# all images in the folder
images_dir = "images"


def main():
    # create a list of all the images
    images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    # get the face encodings for each image
    known_face_encodings = []
    known_face_names = []
    for image in images:
        # load the image
        img = face_recognition.load_image_file(os.path.join(images_dir, image))
        # get the encoding
        encoding = face_recognition.face_encodings(img)[0]
        # add the encoding and the name to the lists
        known_face_encodings.append(encoding)
        known_face_names.append(image.split(".")[0])
    # print the files names in the images folder
    # print(images)
    # print the names of the people in the images folder
    print(known_face_names)

    # load the video
    video = cv2.VideoCapture(video_file)
    # initialize the variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # loop over the frames of the video
    while True:
        # grab the current frame
        ret, frame = video.read()
        # if we are viewing a video, and we did not grab a frame, then we have reached the end of the video
        if not ret:
            break
        # resize the frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # only process every other frame of video to save time
        if process_this_frame:
            # find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # see if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        # display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print("[INFO] Showing video...")
        # display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release handle to the webcam
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
