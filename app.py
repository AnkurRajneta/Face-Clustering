import os
import time
import shutil
import tempfile
import numpy as np
import streamlit as st
import cv2
import face_recognition
from sklearn.cluster import DBSCAN
from imutils import build_montages

flag1 = 0

st.set_page_config(layout='wide')

st.markdown("<h1 style='text-align: center; color: Orange;'>Face Clustering</h1>", unsafe_allow_html=True)

st.text("Please upload the images of multiple faces or faces with different expressions.")
st.text("Wait for the dialog box to confirm successful upload.")

uploaded_files = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
 
no_of_files = len(uploaded_files)

if no_of_files > 0:
    placeholder = st.empty()
    placeholder.success(f"{no_of_files} Images uploaded successfully!")
    time.sleep(3)
    placeholder.empty()
    
    data = []

    for f in uploaded_files:
        # Save the file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        tfile.flush()  # Ensure data is written to disk
        
        # Load the image with error handling
        image = cv2.imread(tfile.name)
        if image is None:
            st.error(f"Failed to load image: {f.name}")
            continue
        
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            st.error(f"Error converting image to RGB: {f.name}")
            continue
        
        # Detect face locations and encodings
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        d = [{"imagePath": tfile.name, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    if len(data) == 0:
        st.warning("No faces detected in the uploaded images.")
    else:
        # Convert the data into a numpy array and get encodings
        data_arr = np.array(data)
        encodings_arr = [item["encoding"] for item in data_arr]

        # Initialize and fit the clustering model on the encoded data
        cluster = DBSCAN(min_samples=3)
        cluster.fit(encodings_arr)

        labelIDs = np.unique(cluster.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])

        st.subheader(f"Number of unique faces identified (excluding unknowns): {numUniqueFaces}")

        if flag1 == 0:
            cols1 = st.columns(numUniqueFaces + 1)
            flag1 = 1

        # Loop over the unique face labels
        for labelID in labelIDs:
            idxs = np.where(cluster.labels_ == labelID)[0]
            idxs = np.random.choice(idxs, size=min(15, len(idxs)), replace=False)

            faces = []
            whole_images = []

            if labelID != -1:
                dir_name = f'face#{labelID + 1}'
                os.mkdir(dir_name)

            for i in idxs:
                current_image = cv2.imread(data_arr[i]["imagePath"])
                rgb_current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
                (top, right, bottom, left) = data_arr[i]["loc"]
                current_face = rgb_current_image[top:bottom, left:right]
                current_face = cv2.resize(current_face, (96, 96))
                
                whole_images.append(rgb_current_image)
                faces.append(current_face)

                if labelID != -1:
                    face_image_name = f'image{i}.jpg'
                    cv2.imwrite(os.path.join(dir_name, face_image_name), current_image)

            if labelID != -1:
                shutil.make_archive(f'zip_face#{labelID + 1}', 'zip', dir_name)
                shutil.rmtree(f'face#{labelID + 1}')

            montage = build_montages(faces, (96, 96), (2, 2))[0]

            current_title = "Face #{}:".format(labelID + 1)
            expander_caption = "Images with Face #{}:".format(labelID + 1)
            current_title = "Unknown:" if labelID == -1 else current_title

            with cols1[labelID + 1]:
                st.write(current_title)
                st.image(montage)
            if labelID != -1:
                with st.expander(expander_caption):
                    cols2 = st.columns(3)
                    for j in range(len(whole_images)):
                        with cols2[j % 3]:
                            st.image(whole_images[j], use_column_width='always')
                    
                    # Provide download button for zip file
                    with open(f"zip_face#{labelID + 1}.zip", "rb") as fp:
                        st.download_button(
                            label=f"Download ZIP of Clustered Images with Face #{labelID + 1}",
                            data=fp,
                            file_name=f"clustered_faces#{labelID + 1}.zip",
                            mime="application/zip"
                        )
                    fp.close()
