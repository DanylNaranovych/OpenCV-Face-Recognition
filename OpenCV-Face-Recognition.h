// CMakeProject1.h : включаемый файл для стандартных системных включаемых файлов
// или включаемые файлы для конкретного проекта.

#pragma once

#include <iostream>
#include <filesystem>
#include <regex>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

constexpr auto PROJECT_DIR = "D:/WorkProjects/OpenCV-Face-Recognition/";
constexpr auto COLLECTED_DIR = "D:/WorkProjects/OpenCV-Face-Recognition/collectedPictures/";
constexpr auto IDENTIFIED_DIR = "D:/WorkProjects/OpenCV-Face-Recognition/identifiedPeople/";
constexpr auto UNIDENTIFIED_DIR = "D:/WorkProjects/OpenCV-Face-Recognition/unidentifiedPeople/";

namespace fs = std::filesystem;

// TODO: установите здесь ссылки на дополнительные заголовки, требующиеся для программы.
