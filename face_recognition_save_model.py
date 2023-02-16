"""Face recognition from webcam using face_recognition package."""
import os
import pickle

import face_recognition as fr


def get_face_encodings(images_path, name):
    """Create face encodings, check whether they are attended to same person and save them to file."""
    dataset = os.listdir(images_path)
    encodings = []
    for i, file in enumerate(dataset):
        print(f"[+] processing {file} {i + 1}/{len(dataset)}")
        image = fr.load_image_file(f"{images_path}{file}")
        face_id = fr.face_encodings(image)[0]

        if len(encodings) == 0:
            encodings.append(face_id)
        else:
            # check whether face is attended to same person
            for item in range(0, len(encodings)):
                result = fr.compare_faces([face_id], encodings[item])
                if result[0]:
                    encodings.append(face_id)
                    print("Same person!")
                    break
                else:
                    print("Another person!")
                    break
    data = {
        "name": name,
        "encodings": encodings,
    }
    # save face encodings to file
    with open(f"{name}_encodings.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] File {name}_encodings.pickle successfully created"


get_face_encodings("images_dataset/arnold/", "Arnold")