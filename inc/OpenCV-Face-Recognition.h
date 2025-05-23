﻿#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <regex>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <sqlite_modern_cpp.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <nlohmann/json.hpp>

using namespace dlib;
using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;

// Path to project, folders and db
extern string PROJECT_DIR, COLLECTED_ENTRY_DIR, COLLECTED_EXIT_DIR, IDENTIFIED_DIR, UNIDENTIFIED_DIR, DB_DIR;

// Global variables for storing the previous descriptor and name
extern matrix<float, 0, 1> lastFaceDescriptor;
extern string lastPersonName;

// Definition of a Dlib neural network
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

struct Config {
	bool showingFrames;
	double thresholdValue;
	string firstCameraMainStream;
	string firstCameraSubStream;
	string secondCameraMainStream;
	string secondCameraSubStream;
};

// Timer. It is necessary for counting down five seconds from the moment
// of saving the last frame before starting the processing of saved photos
class Timer {
public:
	Timer() : startTime(chrono::steady_clock::now()) {}

	// Timer reset
	void reset() {
		startTime = chrono::steady_clock::now();
	}

	// Returns the time in seconds since the last reset or startup
	double elapsed() const {
		auto currentTime = chrono::steady_clock::now();
		chrono::duration<double> elapsedTime = currentTime - startTime;
		return elapsedTime.count();
	}

private:
	chrono::steady_clock::time_point startTime;
};

// Getting last frame index in folder
int getLastFrameNumber(const string& directoryPath, const string& patternPart);
// Analyzing collected frames and making db notes
void processCollectedPictures(bool isEntry);

// Function for wriring logs
void addLog(const string& text);

// Database interaction
// Function for initializing a database
void databaseInitialization();
// Function for adding a new entry with name and entry time
void addRecord(const std::string& name, const string& entryTime);
// Function to update the output time for the last entry by name
void addExitTimeToRecord(const std::string& name, const string& exitTime);
// Function to check if a user has an unclosed record
bool checkOpenRecord(const std::string& name);
// Function to get the id of an open record
int getIdOfOpenRecord(const std::string& name);
