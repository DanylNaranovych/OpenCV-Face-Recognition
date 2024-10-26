#include "../inc/OpenCV-Face-Recognition.h"

std::string getFileCreationTime(const std::string& filePath) {
	// Получаем время последнего изменения файла
	auto ftime = fs::last_write_time(filePath);

	// Преобразуем file_time_type в system_clock::time_point
	auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
		ftime - decltype(ftime)::clock::now() + std::chrono::system_clock::now()
	);

	// Преобразуем в time_t
	std::time_t ctime = std::chrono::system_clock::to_time_t(sctp);

	// Преобразуем в строку с помощью stringstream
	std::stringstream ss;
	ss << std::put_time(std::localtime(&ctime), "%Y-%m-%d %H:%M:%S");

	// Возвращаем строковое представление времени
	return ss.str();
}

void processCollectedPictures() {
	bool isEntry = false, knownPerson = false;
	short int index, collectedIndexEntry, collectedIndexExit;
	size_t identifiedPersonNameSize;
	string personName, identifiedPersonName, filePath;
	cv::Mat img, knownImg;
	matrix<rgb_pixel> face_chip;
	Timer timer;

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	anet_type face_recognizer;
	string pose_model_path = PROJECT_DIR + "models/shape_predictor_68_face_landmarks.dat";
	string face_recognizer_path = PROJECT_DIR + "models/dlib_face_recognition_resnet_model_v1.dat";
	cout << pose_model_path << endl;
	cout << face_recognizer_path << endl;
	deserialize(pose_model_path) >> pose_model;
	deserialize(face_recognizer_path) >> face_recognizer;

	cout << "Processing of collected frames has started" << endl;

	while (true) {
		// Press esc to close the program
		if (GetAsyncKeyState(VK_ESCAPE)) {
			cout << "End processing of collected frames loop" << endl;
			return;
		}

		if (timer.elapsed() > 5) {
			collectedIndexEntry = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame_entry");
			collectedIndexExit = getLastFrameNumber(COLLECTED_DIR, "motion_detected_frame_exit");

			if (collectedIndexEntry >= 0) {
				filePath = COLLECTED_DIR + string("motion_detected_frame_entry_" + to_string(collectedIndexEntry) + ".jpg");
				isEntry = true;
			}
			else if (collectedIndexExit >= 0) {
				filePath = COLLECTED_DIR + string("motion_detected_frame_exit_" + to_string(collectedIndexExit) + ".jpg");
				isEntry = false;
			}
			else {
				cout << "No frames detected. Wait 5 second before another check" << endl;
				timer.reset();
				continue;
			}
		}
		else {
			continue;
		}

		cout << "processingPictures: 1" << endl;

		img = cv::imread(filePath);

		if (img.empty()) {
			cout << "File is empty or unreachable: " << filePath << endl;
			continue;
		}

		cout << "processingPictures: 2" << endl;

		cv_image<bgr_pixel> cimg(img);
		std::vector<rectangle> faces = detector(cimg);

		cout << "processingPictures: 3" << endl;

		// Checking the presence of faces on the frame
		if (!faces.empty()) {
			personName = "unknownPerson";

			cout << "processingPictures: 4" << endl;

			// Loop through all faces in the frame
			for (const auto& face : faces) {

				cout << "processingPictures: 5" << endl;

				// Retrieve the descriptor for the face in the current image
				full_object_detection shape = pose_model(cimg, face);
				cout << "processingPictures: 5.0.1" << endl;

				extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);

				cout << "processingPictures: 5.0.2" << endl;

				matrix<float, 0, 1> faceDescriptor = face_recognizer(face_chip);

				cout << "processingPictures: 5.1" << endl;

				//Check if the current face matches the previously recognized face
				if (!lastPersonName.empty()) {

					cout << "processingPictures: 5.2" << endl;

					double distance = length(faceDescriptor - lastFaceDescriptor);
					// Threshold for identifying similar individuals
					if (distance < 0.6) {

						cout << "processingPictures: 5.3" << endl;

						cout << "The same person detected: " << lastPersonName << endl;
						personName = lastPersonName;
						knownPerson = true;
					}
				}

				cout << "processingPictures: 6" << endl;

				// If the person has not been identified in the past condition,
				// then we start identification relative to all known people
				if (!knownPerson) {

					cout << "processingPictures: 7" << endl;

					// Search the folder of identified people
					for (const auto& personEntry : fs::directory_iterator(IDENTIFIED_DIR)) {
						identifiedPersonName = personEntry.path().filename().string();
						identifiedPersonNameSize = identifiedPersonName.size();
						if (identifiedPersonName.substr(identifiedPersonNameSize - 4) != ".jpg")  continue;

						knownImg = cv::imread(IDENTIFIED_DIR + identifiedPersonName);

						identifiedPersonName.erase(identifiedPersonNameSize - 4);

						cv_image<bgr_pixel> knownCimg(knownImg);
						std::vector<rectangle> knownFaces = detector(knownCimg);

						if (!knownFaces.empty()) {
							full_object_detection knownShape = pose_model(knownCimg, knownFaces[0]);
							extract_image_chip(knownCimg, get_face_chip_details(knownShape, 150, 0.25), face_chip);
							matrix<float, 0, 1> knownFaceDescriptor = face_recognizer(face_chip);

							// Comparing face descriptors using Euclidean distance
							double distance = length(faceDescriptor - knownFaceDescriptor);
							// Threshold for identifying similar individuals
							if (distance < 0.6) {
								knownPerson = true;
								personName = identifiedPersonName;
								lastFaceDescriptor = knownFaceDescriptor; // Сохраняем последний дескриптор
								lastPersonName = personName; // Сохраняем имя последнего человека
								break;
							}
						}
						if (knownPerson) break;
					}
				}
			}

			if (knownPerson) {
				isEntry ? addRecord(personName, getFileCreationTime(filePath)) : addExitTimeToRecord(personName, getFileCreationTime(filePath));

				cout << "The person is identified, photo deletion at path: " << filePath << endl;
				fs::remove(filePath);
			}
			else {
				index = getLastFrameNumber(UNIDENTIFIED_DIR, personName);
				string destination = UNIDENTIFIED_DIR + personName + "_" + to_string(++index) + ".jpg";
				cout << "The person is unidentified, preservation in: " << destination << endl;
				fs::rename(filePath, destination);
			}
		}
		else {
			cout << "No faces detected in the file: " << filePath << endl;
			fs::remove(filePath);
		}
	}
}
