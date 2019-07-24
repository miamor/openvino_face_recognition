## Creating a Gallery for Face Recognition

To recognize faces on a frame, the demo needs a gallery of reference images. Each image should contain a tight crop of face. You can create the gallery from an arbitrary list of images:
1. Put images containing tight crops of frontal-oriented faces to a separate empty folder. Each identity could have multiple images. Name images as `id_name.0.png, id_name.1.png, ...`.
2. Run the `create_list.py <path_to_folder_with_images>` command to get a list of files and identities in `.json` format.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./face_recognition -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

face_recognition [OPTION]
Options:

    -h                             Print a usage message.
    -i '<path>'                    Required. Path to a video or image file. Default value is "cam" to work with camera.
    -m_fd '<path>'                 Required. Path to the Face Detection Retail model (.xml) file.
    -m_lm '<path>'                 Required. Path to the Facial Landmarks Regression Retail model (.xml) file.
    -m_reid '<path>'               Required. Path to the Face Reidentification Retail model (.xml) file.
    -l '<absolute_path>'           Optional. For CPU custom layers, if any. Absolute path to a shared library with the kernels implementation.
          Or
    -c '<absolute_path>'           Optional. For GPU custom kernels, if any. Absolute path to an .xml file with the kernels description.
    -d_fd '<device>'               Optional. Specify the target device for Face Detection Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).
    -d_lm '<device>'               Optional. Specify the target device for Landmarks Regression Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). 
    -d_reid '<device>'             Optional. Specify the target device for Face Reidentification Retail (CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). 
    -out_v  '<path>'               Optional. File to write output video with visualization to.
    -pc                            Optional. Enables per-layer performance statistics.
    -r                             Optional. Output Inference results as raw values.
    -t_ad                          Optional. Probability threshold for person/action detection.
    -t_fd                          Optional. Probability threshold for face detections.
    -inh_fd                        Optional. Input image height for face detector.
    -inw_fd                        Optional. Input image width for face detector.
    -exp_r_fd                      Optional. Expand ratio for bbox before face recognition.
    -t_reid                        Optional. Cosine distance threshold between two vectors for face reidentification.
    -fg                            Optional. Path to a faces gallery in .json format.
    -no_show                       Optional. Do not show processed video.
    -last_frame                    Optional. Last frame number to handle in demo. If negative, handle all input video.
    -crop_gallery                  Optional. Crop images during faces gallery creation.
    -t_reg_fd                      Optional. Probability threshold for face detections during database registration.
    -min_size_fr                   Optional. Minimum input size for faces during database registration.
    -ss_t                          Optional. Number of frames to smooth actions.
    -roi                           Optional. Enable roi
```

Running the application with the empty list of options yields the usage message given above and an error message.

To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO [Model Downloader](https://github.com/opencv/open_model_zoo/tree/master/model_downloader) or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

Example of a valid command line to run the application with pre-trained models for recognizing identities from faces_gallery:
```sh
./face_recognition  -m_fd <path_to_model>/face-detection-adas-0001.xml \
                    -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
                    -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
                    -fg <path_to_faces_gallery.json> \
                    -i <path_to_video>
./face_recognition  -m_fd <path_to_model>/face-detection-adas-0001.xml \
                    -m_reid <path_to_model>/face-reidentification-retail-0095.xml \
                    -m_lm <path_to_model>/landmarks-regression-retail-0009.xml \
                    -fg <path_to_faces_gallery.json> \
                    -i <path_to_video> \
                    -roi
```
