import cv2
import face_recognition
import logging
import numpy as np
import os
import argparse
import sys

parser = argparse.ArgumentParser(description='tutorial ТУТА:')
parser.add_argument('--debug', dest='level', default="INFO", help="This is DEBUUUUG!"
                                                                  "Укажи какой уровень дебага вывести в консоль."
                                                                  "Уровни от самого низшего до высшего: "
                                                                  "DEBUG->INFO->WARNING->ERROR->CRITICAL."
                                                                  " При указании уровня ниже, все сообщения уровня"
                                                                  " выше автоматически подгружаются")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%b-%y %H:%M:%S')

fh = logging.FileHandler('logs.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler(stream=sys.stdout)
numeric_level = getattr(logging, args.level.upper(), None)
ch.setLevel(numeric_level)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

def load_photo():
    pic = []
    names = []
    for root, dirs, files in os.walk(".\people", topdown=False):
        for dir in dirs:
           path = os.path.join(root, dir)
           img = os.listdir(path)
           img = list(map(lambda x: path+'\\'+x, img))
           for i in range(len(img)):
               names.append(os.path.basename(dir))
           pic.extend(img)

    if len(pic) != len(names):
        logger.critical('ERROR! Ошибка при загрузке картинок')

    return pic, names


def draw_name(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        logger.debug("Сделали подпись: %s" % name)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


def draw_parts(i, frame, cords, face_landmarks_list):
    (top, right, bottom, left) = cords
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    logger.debug("Обвели красным прямоугольником")
    test = face_landmarks_list
    pts = np.array(test[i]['top_lip'], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
    pts = np.array(test[i]['bottom_lip'], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)


def main():
    logger.debug("Запустили программу")
    known_faces = []
    pic, names = load_photo()
    cap = cv2.VideoCapture(0)


    i = 0
    for img in pic.copy():
        image = face_recognition.load_image_file(img)
        tmp = face_recognition.face_encodings(image)
        if len(tmp) == 0:
            logger.debug("На фото %s нет лиц" % img)
            pic.pop(i)
            names.pop(i)
        else:
            known_faces.append(tmp[0])
            logger.debug("Обнаружено лицо на фото %s" % img)
            i += 1

    logger.debug("Обработали изображения с лицами")
    logger.debug(pic)
    logger.debug(names)

    while(True):
        #face_locations = []
        #face_encodings = []
        #face_names = []
        ret, frame = cap.read()
        logger.debug("Считали кадр видео")
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        logger.debug("Ищем лица")
        face_locations = face_recognition.face_locations(rgb_frame, model="")
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        for i in range(len(face_locations)):
            draw_parts(i, frame, face_locations[i], face_landmarks_list)
            logger.debug("Нашли лицо")

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            logger.debug("Ищем совпадения с предоставленными лицами")
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.35)  #допуск
            for j in range(len(known_faces)):
                if match[j]:
                    logger.debug("Нашли совпадение: %s" % names[j])
                    face_names.append(names[j])
        draw_name(frame, face_locations, face_names)

        cv2.imshow('Finding people', frame)  # output image

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Нажали q для выхода")
            break

        face_locations.clear()
        face_encodings.clear()
        face_names.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        print("Лошок")
