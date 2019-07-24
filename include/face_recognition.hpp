// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/// @brief Message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief Message for images argument
static const char video_message[] = "Required. Path to a video or image file. Default value is \"cam\" to work with camera.";

/// @brief Message for model argument
static const char face_detection_model_message[] = "Required. Path to the Face Detection Retail model (.xml) file.";
static const char facial_landmarks_model_message[] = "Required. Path to the Facial Landmarks Regression Retail model (.xml) file.";
static const char face_reid_model_message[] = "Required. Path to the Face Reidentification Retail model (.xml) file.";

/// @brief Message for assigning Face Detection inference to device
static const char target_device_message_face_detection[] = "Optional. Specify the target device for Face Detection Retail "\
                                                           "(CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO).";

/// @brief Message for assigning Landmarks Regression retail inference to device
static const char target_device_message_landmarks_regression[] = "Optional. Specify the target device for Landmarks Regression Retail "\
                                                        "(CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). ";

/// @brief Message for assigning Face Reidentification retail inference to device
static const char target_device_message_face_reid[] = "Optional. Specify the target device for Face Reidentification Retail "\
                                                        "(CPU, GPU, FPGA, HDDL, MYRIAD, or HETERO). ";

/// @brief Message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance statistics.";

/// @brief Message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Optional. For GPU custom kernels, if any. "\
"Absolute path to an .xml file with the kernels description.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Optional. For CPU custom layers, if any. " \
"Absolute path to a shared library with the kernels implementation.";

/// @brief Message for probability threshold argument for face detections
static const char face_threshold_output_message[] = "Optional. Probability threshold for face detections.";

/// @brief Message for probability threshold argument for person/action detection
static const char person_threshold_output_message[] = "Optional. Probability threshold for person/action detection.";

/// @brief Message for cosine distance threshold for face reidentification
static const char threshold_output_message_face_reid[] = "Optional. Cosine distance threshold between two vectors for face reidentification.";

/// @brief Message for faces gallery path
static const char reid_gallery_path_message[] = "Optional. Path to a faces gallery in .json format.";

/// @brief Message for output video path
static const char output_video_message[] = "Optional. File to write output video with visualization to.";

/// @brief Message raw output flag
static const char raw_output_message[] = "Optional. Output Inference results as raw values.";

/// @brief Message no show processed video
static const char no_show_processed_video[] = "Optional. Do not show processed video.";

/// @brief Message input image height for face detector
static const char input_image_height_output_message[] = "Optional. Input image height for face detector.";

/// @brief Message input image width for face detector
static const char input_image_width_output_message[] = "Optional. Input image width for face detector.";

/// @brief Message expand ratio for bbox
static const char expand_ratio_output_message[] = "Optional. Expand ratio for bbox before face recognition.";

/// @brief Message last frame number to handle
static const char last_frame_message[] = "Optional. Last frame number to handle in demo. If negative, handle all input video.";

/// @brief Message crop gallery
static const char crop_gallery_message[] = "Optional. Crop images during faces gallery creation.";

/// @brief Message for probability threshold argument for face detections during database registration.
static const char face_threshold_registration_output_message[] = "Optional. Probability threshold for face detections during database registration.";

/// @brief Message for minumum input size for faces database registration.
static const char min_size_fr_reg_output_message[] = "Optional. Minimum input size for faces during database registration.";

/// @brief Message for number of frames for action tracker
static const char tracker_smooth_size_message[] = "Optional. Number of frames to smooth actions.";

/// @brief Message for roi argument
static const char roi_message[] = "Optional. Enable roi";


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// @brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "cam", video_message);

/// @brief Define parameter for face detection model file <br>
/// It is a required parameter
DEFINE_string(m_fd, "", face_detection_model_message);

/// @brief Define parameter for facial landmarks model file <br>
/// It is a required parameter
DEFINE_string(m_lm, "", facial_landmarks_model_message);

/// @brief Define parameter for face reidentification model file <br>
/// It is a required parameter
DEFINE_string(m_reid, "", face_reid_model_message);

/// @brief device the target device for face detection on <br>
DEFINE_string(d_fd, "CPU", target_device_message_face_detection);

/// @brief device the target device for facial landnmarks regression infer on <br>
DEFINE_string(d_lm, "CPU", target_device_message_landmarks_regression);

/// @brief device the target device for face reidentification infer on <br>
DEFINE_string(d_reid, "CPU", target_device_message_face_reid);

/// @brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Flag to output raw pipeline results<br>
/// It is an optional parameter
DEFINE_bool(r, false, raw_output_message);

/// @brief Define probability threshold for person/action detection <br>
/// It is an optional parameter
DEFINE_double(t_ad, 0.4, person_threshold_output_message);

/// @brief Define probability threshold for face detections <br>
/// It is an optional parameter
DEFINE_double(t_fd, 0.6, face_threshold_output_message);

/// @brief Define cosine distance threshold for face reid <br>
/// It is an optional parameter
DEFINE_double(t_reid, 0.7, threshold_output_message_face_reid);

/// @brief Path to a faces gallery for reid <br>
/// It is a optional parameter
DEFINE_string(fg, "", reid_gallery_path_message);

/// @brief File to write output video with visualization to.
/// It is a optional parameter
DEFINE_string(out_v, "", output_video_message);

/// @brief Flag to disable processed video showing<br>
/// It is an optional parameter
DEFINE_bool(no_show, false, no_show_processed_video);

/// @brief Input image height for face detector<br>
/// It is an optional parameter
DEFINE_int32(inh_fd, 600, input_image_height_output_message);

/// @brief Input image width for face detector<br>
/// It is an optional parameter
DEFINE_int32(inw_fd, 600, input_image_width_output_message);

/// @brief Expand ratio for bbox before face recognition<br>
/// It is an optional parameter
DEFINE_double(exp_r_fd, 1.15, face_threshold_output_message);

/// @brief Input image height for face detector<br>
/// It is an optional parameter
DEFINE_int32(last_frame, -1, last_frame_message);

/// @brief Flag to enable image cropping during database creation<br>
/// It is an optional parameter
DEFINE_bool(crop_gallery, false, crop_gallery_message);

/// @brief Define probability threshold for face detections during registration<br>
/// It is an optional parameter
DEFINE_double(t_reg_fd, 0.9, face_threshold_registration_output_message);

/// @brief Minimum input image width & heigh for sucessful face registration<br>
/// It is an optional parameter
DEFINE_int32(min_size_fr, 128, min_size_fr_reg_output_message);

/// @brief Number of frames to smooth actions<br>
/// It is an optional parameter
DEFINE_int32(ss_t, -1, tracker_smooth_size_message);

/// @brief Define a flag to enable roi<br>
/// It is an optional parameter
DEFINE_bool(roi, false, roi_message);


/**
* @brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "face_recognition [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                             " << help_message << std::endl;
    std::cout << "    -i '<path>'                    " << video_message << std::endl;
    std::cout << "    -m_fd '<path>'                 " << face_detection_model_message << std::endl;
    std::cout << "    -m_lm '<path>'                 " << facial_landmarks_model_message << std::endl;
    std::cout << "    -m_reid '<path>'               " << face_reid_model_message << std::endl;
    std::cout << "    -l '<absolute_path>'           " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "    -c '<absolute_path>'           " << custom_cldnn_message << std::endl;
    std::cout << "    -d_fd '<device>'               " << target_device_message_face_detection << std::endl;
    std::cout << "    -d_lm '<device>'               " << target_device_message_landmarks_regression << std::endl;
    std::cout << "    -d_reid '<device>'             " << target_device_message_face_reid << std::endl;
    std::cout << "    -out_v  '<path>'               " << output_video_message << std::endl;
    std::cout << "    -pc                            " << performance_counter_message << std::endl;
    std::cout << "    -r                             " << raw_output_message << std::endl;
    std::cout << "    -t_ad                          " << person_threshold_output_message << std::endl;
    std::cout << "    -t_fd                          " << face_threshold_output_message << std::endl;
    std::cout << "    -inh_fd                        " << input_image_height_output_message << std::endl;
    std::cout << "    -inw_fd                        " << input_image_width_output_message << std::endl;
    std::cout << "    -exp_r_fd                      " << expand_ratio_output_message << std::endl;
    std::cout << "    -t_reid                        " << threshold_output_message_face_reid << std::endl;
    std::cout << "    -fg                            " << reid_gallery_path_message << std::endl;
    std::cout << "    -no_show                       " << no_show_processed_video << std::endl;
    std::cout << "    -last_frame                    " << last_frame_message << std::endl;
    std::cout << "    -crop_gallery                  " << crop_gallery_message << std::endl;
    std::cout << "    -t_reg_fd                      " << face_threshold_registration_output_message << std::endl;
    std::cout << "    -min_size_fr                   " << min_size_fr_reg_output_message << std::endl;
    std::cout << "    -ss_t                          " << tracker_smooth_size_message << std::endl;
    std::cout << "    -roi                           " << roi_message << std::endl;
}
