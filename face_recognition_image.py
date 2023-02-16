"""Face recognition from images using face_recognition package."""

import face_recognition as fr
from PIL import Image, ImageDraw


def face_recognition(image_path):
    """Recognize faces on the image, draw a rectangle around the faces and show the resulting image."""
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)
    print(f"{len(face_locations)} face(s) found.")

    # draw a rectangle around the faces
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline="green", width=3)
    del draw

    return pil_image.show()


def extract_faces(image_path):
    """Extract faces from an image and save them as a separate files."""
    count = 0
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)

    for face in face_locations:
        top, right, bottom, left = face
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(f"media/face_{count}.jpg")
        count += 1

    return f"{len(face_locations)} face(s) found."


def compare_faces(image_1_path, image_2_path):
    """Compare two faces from the images."""
    image_1 = fr.load_image_file(image_1_path)
    image_1_encodings = fr.face_encodings(image_1)[0]

    image_2 = fr.load_image_file(image_2_path)
    image_2_encodings = fr.face_encodings(image_2)[0]

    result = fr.compare_faces([image_1_encodings], image_2_encodings)

    print(result)


face_recognition("media/people.jpg")
print(extract_faces("media/people.jpg"))
compare_faces("media/Ben Affleck.jpg", "media/ben_to compare.jpg")
compare_faces("media/Ben Affleck.jpg", "media/objects.jpg")
