#include "page_dewarp.hpp"
#include "C_preprocess.hpp"
#include <android/log.h>
#include <fstream>
#include <jni.h>
#define JNIIMPORT
#define JNIEXPORT  __attribute__ ((visibility ("default")))
#define JNICALL

using namespace std;
/*int main(int argc, char **argv)
{
    ofstream outfile;
	outfile.open("C_PREPROCESS_RESULT.txt", ofstream::out | ofstream::app);

    float width_threshold = 200;
    float height_threshold = 300;
    cv::Mat dst;
    cv::Mat image;
    Action ac, _ac;
   if (argc == 1)
	{
		return 0;
	}

	string filename(argv[1]);
	if (filename.empty())
	{
		//help();
		cout << "Nhap vao anh" << endl;
		return -1;
	}
	image = cv::imread(filename);
	if (image.empty())
	{
		//help();
		cout << "can not open " << filename << endl;
		return -1;
	}
    //cv::cvtColor(image, dst, cv::COLOR_BGR2GRAY);
    //page_dewarp(image.clone(), dst);
    //C_preprocess(dst, dst, ac);
	std::vector<cv::Point2f> point;
	C_preprocess(image, dst, ac, &point);
	outfile << "Action for " << argv[1] << ": " << ac << endl;
	outfile.close();
    
    // if (_ac == Action::chup_anh) {
		string outfile_prefix(argv[1]);
    	size_t lastIndex = outfile_prefix.find_last_of(".");
		outfile_prefix = outfile_prefix.substr(0, lastIndex);
    	//page_dewarp(image, dst, outfile_prefix);
	// }
	if (point.size() == 4)
		page_dewarp(image, dst, point, outfile_prefix);
	
   
    
    // if (ac == Action::chup_anh)
    //     page_dewarp(image, dst, point);
}*/
extern "C" {

/*JNIEXPORT jint JNICALL Java_com_example_builddewarp_CaptureImage_00024ImageSave_getvalue
        (JNIEnv*, jobject, jlong inpAddr){
    Mat& image = *(Mat*) inpAddr;
	cv::Mat dst;
	Action ac;
	std::vector<cv::Point2f> point;
    C_preprocess(image, dst, ac, point);
    __android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%d", ac);
	return ac;
};*/

JNIEXPORT jlong JNICALL Java_com_example_builddewarp_CaptureImage_00024ImageSave_dewarpImage
        (JNIEnv*, jobject, jlong src, jlong dst){
    Mat& img_src = *(Mat*) src;
    Mat& img_dst = *(Mat*) dst;
    ofstream outfile;
	Mat *mat;
    std::vector<cv::Point2f> point;
    page_dewarp(img_src, img_dst, line_point);
    mat = &img_dst;
    __android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%p", mat);
    __android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%d", mat->cols);
    __android_log_print(ANDROID_LOG_ERROR, "DEBUG_get_page_extent", "%d", mat->rows);
    return (jlong) mat;
};
}