// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <map>
#include <set>
#include <vector>
#include <fstream>

#include "logger.hpp"

namespace {

const char unknown_label[] = "Unknown";

std::string GetUnknownOrLabel(const std::vector<std::string>& labels, int idx)  {
    return idx >= 0 ? labels.at(idx) : unknown_label;
}
}  // anonymous namespace

DetectionsLogger::DetectionsLogger(std::ostream& stream, bool enabled)
    : log_stream_(stream) {
    write_logs_ = enabled;
}

void DetectionsLogger::CreateNextFrameRecord(const std::string& path, const int frame_idx,
                                             const size_t width, const size_t height) {
    if (write_logs_)
        log_stream_ << "Frame_name: " << path << "@" << frame_idx << " width: "
                    << width << " height: " << height << std::endl;
}

void DetectionsLogger::AddFaceToFrame(const cv::Rect& rect, const std::string& id) {
    if (write_logs_) {
        log_stream_ << "Object type: face. Box: " << rect << " id: " << id;
        log_stream_ << std::endl;
    }
}

void DetectionsLogger::FinalizeFrameRecord() {
    if (write_logs_) {
        log_stream_ << std::endl;
    }
}

void DetectionsLogger::DumpDetections(const std::string& video_path,
                                      const cv::Size frame_size,
                                      const size_t num_frames,
                                      const std::vector<Track>& face_tracks,
                                      const std::map<int, int>& track_id_to_label_faces,
                                      const std::vector<std::string>& person_id_to_label)  {
    std::map<int, std::vector<const TrackedObject*>> frame_idx_to_face_track_objs;

    for (const auto& tr : face_tracks) {
        for (const auto& obj : tr.objects) {
            frame_idx_to_face_track_objs[obj.frame_idx].emplace_back(&obj);
        }
    }

    for (size_t i = 0; i < num_frames; i++)  {
        CreateNextFrameRecord(video_path, i, frame_size.width, frame_size.height);

        for (const auto& p_obj : frame_idx_to_face_track_objs[i]) {
            const auto& obj = *p_obj;

            std::string face_label = GetUnknownOrLabel(person_id_to_label, track_id_to_label_faces.at(obj.object_id));
            AddFaceToFrame(obj.rect, face_label);
        }

        FinalizeFrameRecord();
    }
}

DetectionsLogger::~DetectionsLogger() {
}
