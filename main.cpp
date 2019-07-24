// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <ext_list.hpp>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
#include <utility>
#include <ie_iextension.h>

#include "cnn.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "image_grabber.hpp"
#include "logger.hpp"
#include "face_recognition.hpp"

using namespace InferenceEngine;


int xmin = 210; //480; //210;
int ymin = 20;
int xmax = 565; //940; //565;
int ymax = 660;


namespace {

class Visualizer {
private:
    cv::Mat frame_;
    cv::Mat top_persons_;
    const bool enabled_;
    const int num_top_persons_;
    cv::VideoWriter& writer_;
    float rect_scale_x_;
    float rect_scale_y_;
    static int const max_input_width_ = 1920;
    std::string const main_window_name_ = "Smart classroom demo";
    std::string const top_window_name_ = "Top-k students";
    static int const crop_width_ = 128;
    static int const crop_height_ = 320;
    static int const header_size_ = 80;
    static int const margin_size_ = 5;

public:
    Visualizer(bool enabled, cv::VideoWriter& writer, int num_top_persons)
        : enabled_(enabled), num_top_persons_(num_top_persons), writer_(writer),
          rect_scale_x_(0), rect_scale_y_(0) {
        if (!enabled_) {
            return;
        }

        cv::namedWindow(main_window_name_);

        if (num_top_persons_ > 0) {
            cv::namedWindow(top_window_name_);

            CreateTopWindow();
            ClearTopWindow();
        }
    }

    static cv::Size GetOutputSize(const cv::Size& input_size) {
        if (input_size.width > max_input_width_) {
            float ratio = static_cast<float>(input_size.height) / input_size.width;
            return cv::Size(max_input_width_, cvRound(ratio*max_input_width_));
        }
        return input_size;
    }

    void SetFrame(const cv::Mat& frame) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        frame_ = frame.clone();
        rect_scale_x_ = 1;
        rect_scale_y_ = 1;
        cv::Size new_size = GetOutputSize(frame_.size());
        if (new_size != frame_.size()) {
            rect_scale_x_ = static_cast<float>(new_size.height) / frame_.size().height;
            rect_scale_y_ = static_cast<float>(new_size.width) / frame_.size().width;
            cv::resize(frame_, frame_, new_size);
        }
    }

    void Show() const {
        if (enabled_) {
            cv::imshow(main_window_name_, frame_);
        }

        if (writer_.isOpened()) {
            writer_ << frame_;
        }
    }

    void DrawCrop(cv::Rect roi, int id, const cv::Scalar& color) const {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        if (id < 0 || id >= num_top_persons_) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
            roi.x = cvRound(roi.x * rect_scale_x_);
            roi.y = cvRound(roi.y * rect_scale_y_);

            roi.height = cvRound(roi.height * rect_scale_y_);
            roi.width = cvRound(roi.width * rect_scale_x_);
        }

        roi.x = std::max(0, roi.x);
        roi.y = std::max(0, roi.y);
        roi.width = std::min(roi.width, frame_.cols - roi.x);
        roi.height = std::min(roi.height, frame_.rows - roi.y);

        const auto crop_label = std::to_string(id + 1);

        auto frame_crop = frame_(roi).clone();
        cv::resize(frame_crop, frame_crop, cv::Size(crop_width_, crop_height_));

        const int shift = (id + 1) * margin_size_ + id * crop_width_;
        frame_crop.copyTo(top_persons_(cv::Rect(shift, header_size_, crop_width_, crop_height_)));

        cv::imshow(top_window_name_, top_persons_);
    }

    void DrawObject(cv::Rect rect, const std::string& label_to_draw,
                    const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        if (FLAGS_roi) {
            rect.x = rect.x + xmin;
            rect.y = rect.y + ymin;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
                rect.x = cvRound(rect.x * rect_scale_x_);
                rect.y = cvRound(rect.y * rect_scale_y_);

                rect.height = cvRound(rect.height * rect_scale_y_);
                rect.width = cvRound(rect.width * rect_scale_x_);
        }
        cv::rectangle(frame_, rect, bbox_color);

        if (plot_bg && !label_to_draw.empty()) {
            int baseLine = 0;
            const cv::Size label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
                          cv::Point(rect.x + label_size.width, rect.y + baseLine),
                          bbox_color, cv::FILLED);
        }
        if (!label_to_draw.empty()) {
            cv::putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                        text_color, 1, cv::LINE_AA);
        }
    }

    void DrawRect(cv::Rect rect, const cv::Scalar& bbox_color) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
                rect.x = cvRound(rect.x * rect_scale_x_);
                rect.y = cvRound(rect.y * rect_scale_y_);

                rect.height = cvRound(rect.height * rect_scale_y_);
                rect.width = cvRound(rect.width * rect_scale_x_);
        }
        cv::rectangle(frame_, rect, bbox_color);
    }

    void DrawFPS(const float fps, const cv::Scalar& color) {
        if (enabled_ && !writer_.isOpened()) {
            cv::putText(frame_,
                        std::to_string(static_cast<int>(fps)) + " fps",
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2,
                        color, 2, cv::LINE_AA);
        }
    }

    void CreateTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        const int width = margin_size_ * (num_top_persons_ + 1) + crop_width_ * num_top_persons_;
        const int height = header_size_ + crop_height_ + margin_size_;

        top_persons_.create(height, width, CV_8UC3);
    }

    void ClearTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        top_persons_.setTo(cv::Scalar(255, 255, 255));

        for (int i = 0; i < num_top_persons_; ++i) {
            const int shift = (i + 1) * margin_size_ + i * crop_width_;

            cv::rectangle(top_persons_, cv::Point(shift, header_size_),
                          cv::Point(shift + crop_width_, header_size_ + crop_height_),
                          cv::Scalar(128, 128, 128), cv::FILLED);

            const auto label_to_draw = "#" + std::to_string(i + 1);
            int baseLine = 0;
            const auto label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
            const int text_shift = (crop_width_ - label_size.width) / 2;
            cv::putText(top_persons_, label_to_draw,
                        cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        cv::imshow(top_window_name_, top_persons_);
    }

    void Finalize() const {
        if (enabled_) {
            cv::destroyWindow(main_window_name_);

            if (num_top_persons_ > 0) {
                cv::destroyWindow(top_window_name_);
            }
        }

        if (writer_.isOpened()) {
            writer_.release();
        }
    }
};


std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
    std::map<int, int> face_track_id_to_label;
    for (const auto& track : face_tracks) {
        const auto& first_obj = track.first_object;
        // check consistency
        // to receive this consistency for labels
        // use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
        for (const auto& obj : track.objects) {
            SCR_CHECK_EQ(obj.label, first_obj.label);
            SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
        }

        auto cur_obj_id = first_obj.object_id;
        auto cur_label = first_obj.label;
        SCR_CHECK(face_track_id_to_label.count(cur_obj_id) == 0) << " Repeating face tracks";
        face_track_id_to_label[cur_obj_id] = cur_label;
    }
    return face_track_id_to_label;
}

}  // namespace

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    return true;
}


int main(int argc, char* argv[]) {
    try {
        /** This demo covers 4 certain topologies and cannot be generalized **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto video_path = FLAGS_i;
        const auto fd_model_path = FLAGS_m_fd;
        const auto fd_weights_path = fileNameNoExt(FLAGS_m_fd) + ".bin";
        const auto fr_model_path = FLAGS_m_reid;
        const auto fr_weights_path = fileNameNoExt(FLAGS_m_reid) + ".bin";
        const auto lm_model_path = FLAGS_m_lm;
        const auto lm_weights_path = fileNameNoExt(FLAGS_m_lm) + ".bin";

        slog::info << "Reading video '" << video_path << "'" << slog::endl;
        ImageGrabber cap(video_path);
        if (!cap.IsOpened()) {
            slog::err << "Cannot open the video" << slog::endl;
            return 1;
        }

        std::map<std::string, InferencePlugin> plugins_for_devices;
        std::vector<std::string> devices = {FLAGS_d_fd, FLAGS_d_lm,
                                            FLAGS_d_reid};

        for (const auto &device : devices) {
            if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
                continue;
            }
            slog::info << "Loading plugin " << device << slog::endl;
            InferencePlugin plugin = PluginDispatcher().getPluginByDevice(device);
            printPluginVersion(plugin, std::cout);
            /** Load extensions for the CPU plugin **/
            if ((device.find("CPU") != std::string::npos)) {
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            }
            if (device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos) {
                plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
            }
            if (FLAGS_pc)
                plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            plugins_for_devices[device] = plugin;
        }

        // Load face detector
        detection::DetectorConfig face_config(fd_model_path, fd_weights_path);
        face_config.plugin = plugins_for_devices[FLAGS_d_fd];
        face_config.is_async = true;
        face_config.enabled = !fd_model_path.empty();
        face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
        face_config.input_h = FLAGS_inh_fd;
        face_config.input_w = FLAGS_inw_fd;
        face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        detection::FaceDetection face_detector(face_config);

        // Load face detector for face database registration
        detection::DetectorConfig face_registration_det_config(fd_model_path, fd_weights_path);
        face_registration_det_config.plugin = plugins_for_devices[FLAGS_d_fd];
        face_registration_det_config.enabled = !fd_model_path.empty();
        face_registration_det_config.is_async = false;
        face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
        face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        detection::FaceDetection face_detector_for_registration(face_registration_det_config);

        // Load face reid
        CnnConfig reid_config(fr_model_path, fr_weights_path);
        reid_config.max_batch_size = 16;
        reid_config.enabled = face_config.enabled && !fr_model_path.empty() && !lm_model_path.empty();
        reid_config.plugin = plugins_for_devices[FLAGS_d_reid];
        VectorCNN face_reid(reid_config);

        // Load landmarks detector
        CnnConfig landmarks_config(lm_model_path, lm_weights_path);
        landmarks_config.max_batch_size = 16;
        landmarks_config.enabled = face_config.enabled && reid_config.enabled && !lm_model_path.empty();
        landmarks_config.plugin = plugins_for_devices[FLAGS_d_lm];
        VectorCNN landmarks_detector(landmarks_config);

        // Create face gallery
        EmbeddingsGallery face_gallery(FLAGS_fg, FLAGS_t_reid, FLAGS_min_size_fr, FLAGS_crop_gallery,
                                       face_detector_for_registration, landmarks_detector, face_reid);

        if (!reid_config.enabled) {
            slog::warn << "Face recognition models are disabled!"  << slog::endl;
        } else if (!face_gallery.size()) {
            slog::warn << "Face reid gallery is empty!"  << slog::endl;
        } else {
            slog::info << "Face reid gallery size: " << face_gallery.size() << slog::endl;
        }

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 1;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8f;
        tracker_reid_params.averaging_window_size_for_rects = 1;
        tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = false;
        tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_reid_params.objects_type = "face";

        Tracker tracker_reid(tracker_reid_params);

        cv::Mat frame, prev_frame, prev_frame_sm;
        detection::DetectedObjects faces;

        float work_time_ms = 0.f;
        size_t work_num_frames = 0;
        size_t total_num_frames = 0;
        const char ESC_KEY = 27;
        const cv::Scalar blue_color(255, 0, 0);
        const cv::Scalar green_color(0, 255, 0);
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);

        if (cap.GrabNext()) {
            cap.Retrieve(frame);
        } else {
            slog::err << "Can't read the first frame" << slog::endl;
            return 1;
        }


        cv::Mat frame_sm;
        // crop region before parse to detector
        cv::Rect myROI(xmin, ymin, xmax-xmin, ymax-ymin);
        if (FLAGS_roi) {
            frame_sm = frame(myROI);
        } else {
            frame_sm = frame;
        }

            face_detector.enqueue(frame_sm);
            face_detector.submitRequest();


        prev_frame = frame.clone();
        prev_frame_sm = frame_sm.clone();

        bool is_last_frame = false;
        auto prev_frame_path = cap.GetVideoPath();

        cv::VideoWriter vid_writer;
        if (!FLAGS_out_v.empty()) {
            vid_writer = cv::VideoWriter(FLAGS_out_v, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         cap.GetFPS(), Visualizer::GetOutputSize(frame.size()));
        }
        Visualizer sc_visualizer(!FLAGS_no_show, vid_writer, -1);
        DetectionsLogger logger(std::cout, FLAGS_r);

        if (!FLAGS_no_show) {
            std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
        }
        while (!is_last_frame) {
            logger.CreateNextFrameRecord(cap.GetVideoPath(), work_num_frames, prev_frame.cols, prev_frame.rows);
            auto started = std::chrono::high_resolution_clock::now();

            is_last_frame = !cap.GrabNext();
            if (!is_last_frame)
                cap.Retrieve(frame);

            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }

            sc_visualizer.SetFrame(prev_frame);


                face_detector.wait();
                face_detector.fetchResults();
                faces = face_detector.results;

                if (!is_last_frame) {
                    prev_frame_path = cap.GetVideoPath();

                    if (FLAGS_roi) {
                        frame_sm = frame(myROI);
                    } else {
                        frame_sm = frame;
                    }

                    face_detector.enqueue(frame_sm);
                    face_detector.submitRequest();
                }

                std::vector<cv::Mat> face_rois, landmarks, embeddings;
                TrackedObjects tracked_face_objects;

                for (const auto& face : faces) {
                    face_rois.push_back(prev_frame_sm(face.rect));
                }
                landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
                AlignFaces(&face_rois, &landmarks);
                face_reid.Compute(face_rois, &embeddings);
                auto ids = face_gallery.GetIDsByEmbeddings(embeddings);

                for (size_t i = 0; i < faces.size(); i++) {
                    int label = ids.empty() ? EmbeddingsGallery::unknown_id : ids[i];
                    tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, label);
                }
                tracker_reid.Process(prev_frame_sm, tracked_face_objects, work_num_frames);

                const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();


                auto elapsed = std::chrono::high_resolution_clock::now() - started;
                auto elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

                work_time_ms += elapsed_ms;

                for (size_t j = 0; j < tracked_faces.size(); j++) {
                    const auto& face = tracked_faces[j];
                    std::string label_to_draw;
                    if (face.label != EmbeddingsGallery::unknown_id)
                        label_to_draw += face_gallery.GetLabelByID(face.label);
                    // else
                    //     label_to_draw += "Unknown"

                    sc_visualizer.DrawObject(face.rect, label_to_draw, white_color, red_color, true);
                    logger.AddFaceToFrame(face.rect, face_gallery.GetLabelByID(face.label));
                }
                
                sc_visualizer.DrawFPS(1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f),
                                      blue_color);

                ++work_num_frames;

            
            if (FLAGS_roi) {
                // draw region of interest
                sc_visualizer.DrawRect(myROI, blue_color);
            }

            ++total_num_frames;

            sc_visualizer.Show();

            if (FLAGS_last_frame >= 0 && work_num_frames > static_cast<size_t>(FLAGS_last_frame)) {
                break;
            }
            prev_frame = frame.clone();
            prev_frame_sm = frame_sm.clone();
            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();

        slog::info << slog::endl;
        if (work_num_frames > 0) {
            const float mean_time_ms = work_time_ms / static_cast<float>(work_num_frames);
            slog::info << "Mean FPS: " << 1e3f / mean_time_ms << slog::endl;
        }
        slog::info << "Frames processed: " << total_num_frames << slog::endl;
        if (FLAGS_pc) {
            face_detector.wait();
            face_detector.PrintPerformanceCounts();
            face_reid.PrintPerformanceCounts();
            landmarks_detector.PrintPerformanceCounts();
        }


            auto face_tracks = tracker_reid.vector_tracks();

            // correct labels for track
            std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_tracks);
            std::map<int, int> face_track_id_to_label = GetMapFaceTrackIdToLabel(new_face_tracks);

    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;

    return 0;
}
