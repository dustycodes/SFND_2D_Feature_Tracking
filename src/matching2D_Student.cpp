#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        int k = 2;
        // Specified in MP.8
        const float ratio_thresh = 0.8f;

        std::vector< std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, k);
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            //-- Filter matches using the Lowe's ratio test
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance)
            {
                matches.push_back(knnMatches[i][0]);
            }
        }

        std::cout << "Keypoints removed: " << knnMatches.size() - matches.size() << std::endl;
    }
}

// Use one of several types of state -of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool useOrientation = false;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, useOrientation);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormed = true;
        bool scaleNormed = true;
        float patternScale = 22.0f;
        int octaves = 4;
        std::vector<int> selectedPairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormed, scaleNormed, patternScale, octaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;

        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    else
    {
        std::cerr << "Failed to find descriptor type " << descriptorType << std::endl;
        return std::numeric_limits<double>::max();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Mat output, output_norm, output_norm_scaled;
    float blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int cornerThresh = 100;

    auto t = (double)cv::getTickCount();

    // Detecting corners
    output = cv::Mat::zeros(img.size(), CV_32FC1);
    cornerHarris(img, output, 2, 3, 0.04);

    // Normalizing
    normalize(output, output_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    convertScaleAbs(output_norm, output_norm_scaled);

    // Drawing a circle around corners
    for(int j = 0; j < output_norm.rows ; j++){
        for(int i = 0; i < output_norm.cols; i++){
            if((int) output_norm.at<float>(j,i) > cornerThresh){
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = blockSize;
                keypoints.push_back(newKeyPoint);
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Apply corner detection
    double t = (double)cv::getTickCount();
    int threshold = 4;
    bool nonmaxSuppression = false;
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::DetectorType::TYPE_5_8;

    cv::FAST(img, keypoints, threshold, nonmaxSuppression, type);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Fast Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    int thresh = 60;
    int octaves = 4;
    float patternScales = 1.0f;
    cv::InputArray mask = cv::noArray();

    auto brisk = cv::BRISK::create(thresh, octaves, patternScales);

    brisk->detect(img, keypoints, mask);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Brisk Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    int nFeatures = 500;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
    cv::InputArray mask = cv::noArray();

    auto orb = cv::ORB::create(nFeatures, scaleFactor, nLevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    orb->detect(img, keypoints, mask);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Orb Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double  detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();

    auto mode = cv::AKAZE::DescriptorType::DESCRIPTOR_MLDB;
    int descriptorSize = 0;
    int descriptorChannels = 3;
    float threshold = 0.001f;
    int octaves = 4;
    int octaveLayers = 4;
    auto diffusivity = cv::KAZE::DIFF_PM_G2;
    cv::InputArray mask = cv::noArray();

    auto akaze = cv::AKAZE::create(mode, descriptorSize, descriptorChannels, threshold, octaves, octaveLayers, diffusivity);
    akaze->detect(img, keypoints, mask);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Akaze Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}

double detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    double t = (double)cv::getTickCount();
    auto detector = cv::xfeatures2d::SIFT::create();
    detector->detect(img, keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Fast detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        std::cout << "Number of keypoints " << keypoints.size() << std::endl;

        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Sift Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}