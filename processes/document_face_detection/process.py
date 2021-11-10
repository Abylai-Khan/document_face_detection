from pathlib import Path

import numpy
from podder_task_foundation import Context, Payload
from podder_task_foundation import Process as ProcessBase
import cv2
from mtcnn.mtcnn import MTCNN


class Process(ProcessBase):
    def initialize(self, context: Context) -> None:
        # Initialization need to be done here
        # Never do the initialization on execute method !
        # - Model loading
        # - Large text(json) file loading
        # - Prepare some data
        # You can get "yourmodel.pth"
        self.detector = MTCNN()

    def execute(self, input_payload: Payload, output_payload: Payload,
        context: Context):
        input_images = input_payload.all(object_type="image")
        face_extend_ratio = 0.2
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        for image_index, input_image in enumerate(input_images):
            input_image._lazy_load()
            pil_image = input_image.get()
            if pil_image.mode is not "RGB":
                pil_image = pil_image.convert("RGB")
            open_cv_image = numpy.array(pil_image)
            image_height, image_width, _ = open_cv_image.shape
            detected_faces = self.detector.detect_faces(open_cv_image)
            face_bbox_list = []
            for detected_face in detected_faces:
                face_left, face_top = tuple(detected_face['box'][0:2])
                face_right, face_bottom = tuple(
                    numpy.array(detected_face['box'][0:2]) + numpy.array(detected_face['box'][2:4]))
                facial_image_size = (face_right - face_left) * (face_bottom - face_top)
                face_bbox_list.append(
                    [face_left, face_top, face_right, face_bottom, facial_image_size])

            if len(face_bbox_list) == 0:
                print(image_index)
                continue

            face_bbox = sorted(face_bbox_list, reverse=True, key=lambda x: x[-1])[0][:-1]
            face_width = face_bbox[2] - face_bbox[0]
            face_height = face_bbox[3] - face_bbox[1]
            face_left = max(0, round(face_bbox[0] - face_width * face_extend_ratio))
            face_top = max(0, round(face_bbox[1] - face_height * face_extend_ratio))
            face_right = min(image_width, round(face_bbox[2] + face_width * face_extend_ratio))
            face_bottom = min(image_height, round(face_bbox[3] + face_height * face_extend_ratio))

            image_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)[1]
            image_binary = cv2.bitwise_not(image_binary)
            image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
            image_binary[face_top:face_bottom, face_left:face_right] = 255

            contours = list(cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0])

            contours.sort(key=lambda x: cv2.contourArea(x), reverse=True)

            facial_image = None
            for contour in contours:
                contour_left, contour_top, contour_width, contour_height = cv2.boundingRect(contour)
                if face_bbox[0] in numpy.arange(contour_left, contour_left + contour_width) and \
                    face_bbox[1] in numpy.arange(contour_top, contour_top + contour_height):
                    facial_image = open_cv_image[contour_top:contour_top + contour_height,
                                   contour_left:contour_left + contour_width].copy()
                    break
            if facial_image is not None:
                cv2.imwrite('./output/document_face_detection/' + "image_" + str(image_index) + ".png", facial_image)
            else:
                print(image_index)
