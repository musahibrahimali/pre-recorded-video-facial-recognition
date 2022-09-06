# import the needed libraries
import face_recognition
import cv2
import os
import numpy as np
import pandas
import pandas as pd
from datetime import date, datetime

# recognize faces in a video recorded video
video_file = "videos/video.mp4"
# all images in the folder
images_dir = "images"
# attendance csv file
attendance_file_csv = "attendance.csv"
attendance_file_excel = "attendance.xlsx"
# absent students file
absent_students_file_csv = "absent_students.csv"
absent_students_file_excel = "absent_students.xlsx"

# directory for unknown students
unknown_students_dir = "unknown_students"


# get the date of the current day
def get_date():
    today = date.today()
    # dd/mm/YY
    d1 = today.strftime("%d/%m/%Y")
    return d1


# get the current time
def get_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


# separate the index number from the name of the student
def separate_index_number(name):
    _name_split = name.split('_')
    # if the length of the split is 2, then return the index number and name, else return a none and the name
    if len(_name_split) == 2:
        return _name_split[0], _name_split[1]
    else:
        return None, name


# write students to file
def write_csv(students: set, file_name: str, isAbsent: bool):
    # print the names in the attendance set
    # print(f"Students Present {students_present}")
    # titles of CSV columns
    fields = ['Index No', 'Name', 'Date', 'Time', 'Present']
    # determine if the student is present or absent
    present = True if not isAbsent else False
    # write the attendance to the csv file
    with open(file_name, "a") as file:
        # write the titles if only the file is empty
        if os.stat(file_name).st_size == 0:
            file.write(",".join(fields) + "\n")
        for student in students:
            # get the index number and name of the student
            # print(f"the student {student}")
            index_number, name = separate_index_number(student)
            # print(f"name {name} index number {index_number}")
            file.write(f"{index_number}, {name}, {get_date()}, {get_time()}, {present}\n")

    # close the students present file
    file.close()


# write to excel file
def write_excel(students: set, file_name: str, isAbsent: bool):
    # title columns
    # titles of CSV columns
    fields = ['Index No', 'Name', 'Date', 'Time', 'Present']
    # create a dataframe and use the fields as the column headings
    dataFrame = pd.DataFrame(columns=fields)
    # determine if the student is present or absent
    present = True if not isAbsent else False
    # loop through the students
    for student in students:
        # create a data frame for each student
        # get the index number and name of the student
        index_number, name = separate_index_number(student)
        # create a dataframe for the student
        student_df = pd.DataFrame([[index_number, name, get_date(), get_time(), present]], columns=fields)
        # append the student dataframe to the main dataframe
        dataFrame = pandas.concat(
            objs=[dataFrame, student_df],
            ignore_index=True,
            axis=0
        )
        # dataFrame = dataFrame.append(student_df, ignore_index=True)
    # write the dataframe to the excel file
    dataFrame.to_excel(file_name, index=False)


def main():
    # create a set to hold the name of students present in the class
    students_present = set()
    # create a set to hold the students who were absent from class
    students_absent = set()
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
    # print(known_face_names)

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
        # save the small frame to a file
        # cv2.imwrite("small_frame.jpg", small_frame)
        # convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # only process every other frame of video to save time
        if process_this_frame:
            # find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
            )
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
                else:
                    # generate a random number to use as the name of the unknown student
                    random_number = np.random.randint(100000, 999999)
                    _file_path = str(unknown_students_dir) + "/" + str(random_number) + "_unknown.jpg"
                    # save the small frame to the unknown students directory
                    cv2.imwrite("./unknown/" + str(str(random_number) + "_unknown.jpg"), small_frame)
                face_names.append(name)
                # add the name of the user to the set
                students_present.add(name)

        process_this_frame = not process_this_frame
        # display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # separate the index number from the name
            _, _name = separate_index_number(name)
            # scale back up face locations since the frame we detected in was scaled to 1/4 size
            # if the name is unknown
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, _name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # print("[INFO] Showing video...")
        # display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release handle to the webcam
    video.release()
    # close the window for the video
    cv2.destroyAllWindows()
    # write_csv(
    #     students=students_present,
    #     file_name=attendance_file,
    #     isAbsent=False
    # )

    write_excel(
        students=students_present,
        file_name=attendance_file_excel,
        isAbsent=False
    )

    # convert the students present to an array
    _students_present = list(students_present)
    _all_students = [f.split(".")[0] for f in images]
    # print(f"Students Present {_students_present}")
    for student in _all_students:
        # print(f"{student in _students_present}")
        # add the student to the set of absent students
        if student not in _students_present:
            students_absent.add(student)

    # print(f"students absent {students_absent}")
    # write the absent students to the file
    # write_csv(
    #     students=students_absent,
    #     file_name=absent_students_file,
    #     isAbsent=True
    # )

    # write to excel
    write_excel(
        students=students_absent,
        file_name=absent_students_file_excel,
        isAbsent=True
    )


if __name__ == "__main__":
    main()
