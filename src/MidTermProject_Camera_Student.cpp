/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgcodecs.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

std::vector<std::string> DETECTOR_TYPES = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
std::vector<std::string> DESCRIPTOR_TYPES = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
std::vector<std::string> MATCHER_TYPES = {"MAT_BF", "MAT_FLANN"};
std::vector<std::string> DESCRIPTOR_CATEGORIES = {"DES_BINARY", "DES_HOG"};
std::vector<std::string> SELECTOR_TYPES = {"SEL_NN", "SEL_KNN"};

struct CombinationStatistic
{
    std::string combination;
    std::vector<uint> numberOfKeyPoints;
    uint totalKeyPoints = 0;
    std::vector<uint> numberOfMatchedKeyPoints;
    uint totalMatches = 0;
    std::vector<double> times;
    double totalTime = 0.0;
    bool status = true;
};

void writeStatsToCSV(const std::vector<CombinationStatistic>& statistics)
{
    string filePath = "../reports/Data.csv";

    std::ofstream fileStream;
    fileStream.open(filePath);

    // Write in CSV format
    // Header
    fileStream << "Combination, Status, Total KeyPoints, Total Matches, Total Time, ";
    size_t numberSamples = 0;
    size_t numberMatches = 0;
    size_t numberTimes = 0;
    for (auto& stat : statistics)
    {
        numberSamples = std::max(numberSamples, stat.numberOfKeyPoints.size());
        numberMatches = std::max(numberMatches, stat.numberOfMatchedKeyPoints.size());
        numberTimes = std::max(numberTimes, stat.times.size());
    }
    for (int i = 0; i < numberSamples; ++i)
    {
        fileStream << "Key Points in Img" << i << ", ";
    }
    for (int i = 0; i < numberMatches; ++i)
    {
        fileStream << "Matches in Img" << i << ", ";
    }
    for (int i = 0; i < numberTimes; ++i)
    {
        fileStream << "Time Img" << i << ", ";
    }

    // Data
    for (auto& stat : statistics)
    {
        fileStream << stat.combination << ", ";
        fileStream << stat.status << ", ";
        fileStream << stat.totalKeyPoints << ", ";
        fileStream << stat.totalMatches << ", ";
        fileStream << stat.totalTime;

        for (auto& keyPointCount : stat.numberOfKeyPoints)
        {
            fileStream << ", " << keyPointCount;
        }
        for (auto& matchCount : stat.numberOfMatchedKeyPoints)
        {
            fileStream << ", " << matchCount;
        }
        for (auto& time : stat.times)
        {
            fileStream << ", " << time;
        }
        fileStream << "\n";
    }

    fileStream.close();
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    bool bVis = false;            // visualize results

    // generate all possible configs
    std::vector<std::vector<std::string>> configOptions = {
            DETECTOR_TYPES, DESCRIPTOR_TYPES, MATCHER_TYPES, DESCRIPTOR_CATEGORIES, SELECTOR_TYPES
    };
    std::vector<CombinationStatistic> statistics;

    /* MAIN LOOP OVER ALL IMAGES */
    // Counters
    int detectorTypeIndex = 0;
    int descriptorTypeIndex = 0;
    int matcherTypeIndex = 0;
    int descriptorCategoryIndex = 0;
    int selectorTypeIndex = 0;
    while (true)
    {
        CombinationStatistic comboStat;

        try
        {
            // Set configs
            string detectorType = DETECTOR_TYPES[detectorTypeIndex];
            string descriptorType = DESCRIPTOR_TYPES[descriptorTypeIndex];
            string matcherType = MATCHER_TYPES[matcherTypeIndex];
            string selectorType = SELECTOR_TYPES[selectorTypeIndex];

            string descriptorCategory = DESCRIPTOR_CATEGORIES[descriptorCategoryIndex];

            std::cout << "Configuration {" << detectorType << ", " << descriptorType << ", " << matcherType << ", "
                      << descriptorCategory << ", " << selectorType << "}" << std::endl;
            std::cout << "Index {" << detectorTypeIndex << ", " << descriptorTypeIndex << ", " << matcherTypeIndex
                      << ", "
                      << descriptorCategoryIndex << ", " << selectorTypeIndex << "}" << std::endl;
            comboStat.combination =
                    detectorType + " " + descriptorType + " " + matcherType + " " + descriptorCategory + " " +
                    selectorType;

            // Debugging helpers
            bool bLimitKpts = false;
            bool bFocusOnVehicle = true;
            cv::Rect vehicleRect(535, 180, 180, 150);
            int maxKeypoints = 50;
            vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame);

                if (dataBuffer.size() > dataBufferSize)
                {
                    dataBuffer.erase(dataBuffer.end());
                }

                //// EOF STUDENT ASSIGNMENT
                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
                double executionTime = 0;
                if (detectorType.compare("SHITOMASI") == 0)
                {
                    executionTime += detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    executionTime += detKeypointsHarris(keypoints, imgGray, false);
                }
                else if (detectorType.compare("FAST") == 0)
                {
                    executionTime += detKeypointsFast(keypoints, imgGray, false);
                }
                else if (detectorType.compare("BRISK") == 0)
                {
                    executionTime += detKeypointsBrisk(keypoints, imgGray, false);
                }
                else if (detectorType.compare("ORB") == 0)
                {
                    executionTime += detKeypointsOrb(keypoints, imgGray, false);
                }
                else if (detectorType.compare("AKAZE") == 0)
                {
                    executionTime += detKeypointsAkaze(keypoints, imgGray, false);
                }
                else if (detectorType.compare("SIFT") == 0)
                {
                    executionTime += detKeypointsSift(keypoints, imgGray, false);
                }
                else
                {
                    std::cerr << "No detector" << std::endl;
                    return -1;
                }

                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                if (bFocusOnVehicle)
                {
                    std::cout << "Number of keypoints " << keypoints.size() << std::endl;

                    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [&](
                            const cv::KeyPoint& keyPoint) {
                        return not vehicleRect.contains(keyPoint.pt);
                    }), keypoints.end());

                    std::cout << "Focused Number of keypoints " << keypoints.size() << std::endl;
                }

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                if (bLimitKpts)
                {
                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;

                executionTime += descKeypoints((dataBuffer.end() - 1)->keypoints
                                               , (dataBuffer.end() - 1)->cameraImg
                                               , descriptors
                                               , descriptorType);
                comboStat.numberOfKeyPoints.push_back(keypoints.size());
                comboStat.totalKeyPoints += keypoints.size();
                comboStat.times.push_back(executionTime);
                comboStat.totalTime += executionTime;
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints
                                     , (dataBuffer.end() - 1)->keypoints
                                     , (dataBuffer.end() - 2)->descriptors
                                     , (dataBuffer.end() - 1)->descriptors
                                     , matches
                                     , descriptorCategory
                                     , matcherType
                                     , selectorType);
                    comboStat.numberOfMatchedKeyPoints.push_back(matches.size());
                    comboStat.totalMatches += matches.size();

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg
                                        , (dataBuffer.end() - 2)->keypoints
                                        , (dataBuffer.end() - 1)->cameraImg
                                        , (dataBuffer.end() - 1)->keypoints
                                        , matches
                                        , matchImg
                                        , cv::Scalar::all(-1)
                                        , cv::Scalar::all(-1)
                                        , vector<char>()
                                        , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }
            } // eof loop over all images
        }
        catch (...)
        {
            std::cerr << "Failed to run configuration: " << comboStat.combination << std::endl;
            comboStat.status = false;
        }

        // Move to next combo of configs
        ++detectorTypeIndex;
        if (detectorTypeIndex >= DETECTOR_TYPES.size())
        {
            ++descriptorTypeIndex;
            detectorTypeIndex = 0;
        }
        if (descriptorTypeIndex >= DESCRIPTOR_TYPES.size())
        {
            ++matcherTypeIndex;
            descriptorTypeIndex = 0;
        }
        if (matcherTypeIndex >= MATCHER_TYPES.size())
        {
            ++selectorTypeIndex;
            matcherTypeIndex = 0;
        }
        if (selectorTypeIndex >= SELECTOR_TYPES.size())
        {
            std::cout << "Finished " << std::endl;
            break;
        }
//        if (selectorTypeIndex >= SELECTOR_TYPES.size())
//        {
//            ++descriptorCategoryIndex;
//            selectorTypeIndex = 0;
//        }
//        if (descriptorCategoryIndex >= DESCRIPTOR_CATEGORIES.size())
//        {
//            std::cout << "Finished " << std::endl;
//            break;
//        }

        statistics.push_back(comboStat);
    }

    writeStatsToCSV(statistics);

    return 0;
}
