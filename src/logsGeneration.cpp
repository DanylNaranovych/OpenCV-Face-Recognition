#include "../inc/OpenCV-Face-Recognition.h"

void addLog(const string& text) {
    ofstream outFile(PROJECT_DIR + "logs.txt", ios::app);

    if (outFile.is_open()) {
        outFile << text;
        outFile.close();
    }
    else {
        cerr << "Failed to open the logs file.\n" << endl;
    }
}
