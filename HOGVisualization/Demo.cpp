#include "fHOG.h"

int main()
{
	fHOG HOG;
	cv::Mat src = cv::imread("face.jpg");

	fHOG::mMat_D_3D feat;
	int res_extract = HOG.extract_feature(src, 8, feat);
	int res_visual = HOG.feature_visualize(feat);
	cv::waitKey(0);

	return res_extract||res_visual;
}