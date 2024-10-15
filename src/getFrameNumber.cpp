#include "../inc/OpenCV-Face-Recognition.h"

int getLastFrameNumber(const string& directoryPath, const string& patternPart) {
	regex filePattern(format(R"({}_(\d+))", patternPart));
	string filename;
	smatch match;
	int lastFrameNumber = -1;

	for (const auto& entry : fs::directory_iterator(directoryPath)) {
		if (entry.is_regular_file()) {
			filename = entry.path().filename().string();
			if (regex_search(filename, match, filePattern)) {
				int frameNumber = stoi(match[1].str());
				if (frameNumber > lastFrameNumber) {
					lastFrameNumber = frameNumber;
				}
			}
		}
	}

	return lastFrameNumber;
}
