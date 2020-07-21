#coding=utf-8
#人脸识别类 - 使用face_recognition模块
import face_recognition

path = "img/face_recognition/"  # 模型数据图片目录
# Load the jpg files into numpy arrays
gates_image = face_recognition.load_image_file(path + "gates.png")
gates1_image = face_recognition.load_image_file(path + "gates1.jpg")
gates2_image = face_recognition.load_image_file(path + "gates2.png")
tangmq_image = face_recognition.load_image_file(path + "tangmq.jpg")
tangmq_test_image = face_recognition.load_image_file(path + "tangmq-test.jpg")
meinv_image = face_recognition.load_image_file(path + "meinv.png")
unknown_image = face_recognition.load_image_file(path + "unknown.png")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    gates_face_encoding = face_recognition.face_encodings(gates_image)[0]
    # print(gates_face_encoding)
    gates1_face_encoding = face_recognition.face_encodings(gates1_image)[0]
    # print(gates1_face_encoding)
    gates2_face_encoding = face_recognition.face_encodings(gates2_image)[0]
    # print(gates2_face_encoding)
    meinv_face_encoding = face_recognition.face_encodings(meinv_image)[0]
    # print(meinv_face_encoding)
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    # print(unknown_face_encoding)
    tangmq_encoding = face_recognition.face_encodings(tangmq_image)[0]
    # print(tangmq_encoding)
    tangmq_test_encoding = face_recognition.face_encodings(tangmq_test_image)[0]
    # print(tangmq_test_encoding)
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    gates_face_encoding,
    meinv_face_encoding,
    tangmq_encoding
]

results1 = face_recognition.compare_faces(known_faces, gates1_face_encoding, tolerance=0.5)
face_distances = face_recognition.face_distance(known_faces, gates1_face_encoding)
print("gates1.jpg 是Bill gates吗? {}".format(results1[0]))
print("gates1.jpg 是Meinv吗? {}".format(results1[1]))
print("gates1.jpg 是未识别的新人吗? {}".format(not True in results1))
print("gates1.jpg 与照片库的距离（越远，置信度越低）： {}" . format(face_distances))

results2 = face_recognition.compare_faces(known_faces, gates2_face_encoding)

print("gates2.png 是Bill gates吗? {}".format(results2[0]))
print("gates2.png 是Meinv吗? {}".format(results2[1]))
print("gates2.png 是未识别的新人吗? {}".format(not True in results2))

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results3 = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("unknown.png 是Bill gates吗? {}".format(results3[0]))
print("unknown.png 是Meinv吗? {}".format(results3[1]))
print("unknown.png 是未识别的新人吗? {}".format(not True in results3))

results4 = face_recognition.compare_faces(known_faces, tangmq_test_encoding, tolerance=0.5)
print("tangmq-test.jpg 是Bill gates吗? {}".format(results4[0]))
print("tangmq-test.jpg 是Meinv吗? {}".format(results4[1]))
print("tangmq-test.jpg 是Tangmq吗? {}".format(results4[2]))
print("tangmq-test.jpg 是未识别的新人吗? {}".format(not True in results4))
