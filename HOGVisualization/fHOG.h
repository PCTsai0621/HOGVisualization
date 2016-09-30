#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>

#include <iostream>
#include <fstream>
#include <time.h>

class fHOG
{
public:
	/*struct of 3D double matrix*/
	struct mMat_D_3D
	{
		int dims0;	//width
		int dims1;	//height
		int dims2;	//depth
		double *data;
		mMat_D_3D(){};
		mMat_D_3D(int _dims0, int _dims1, int _dims2, double* _data)
		{
			dims0 = _dims0;
			dims1 = _dims1;
			dims2 = _dims2;
			data = (double*)malloc(dims0*dims1*dims2*sizeof(double));
			for (size_t k = 0; k < dims2; k++)
				for (size_t i = 0; i < dims1; i++)
					for (size_t j = 0; j < dims0; j++)
						*(data + k*dims0*dims1 + i*dims0 + j) = *(_data + k*dims0*dims1 + i*dims0 + j);
		};
	};

private:
	static const double  EPS;				// small value, used to avoid division by zero
	static const double  uu[9];				// project unit vectors used to compute gradient orientation
	static const double  vv[9]; 			// project unit vectors used to compute gradient orientation
	static const double  uu_f[9];			// the unit vector of Flow vector corresponding to gradient orientation
	static const double  vv_f[9]; 			// the unit vector of Flow vector corresponding to gradient orientation

private:
	//***Comparison***//

	/*minimal comparison*/
	inline double min(double _In1, double _In2) { return (_In1 <= _In2 ? _In1 : _In2); };
	/*minimal comparison*/
	inline int min(int _In1, int _In2) { return (_In1 <= _In2 ? _In1 : _In2); };
	/*maximal comparison*/
	inline double max(double _In1, double _In2) { return (_In1 <= _In2 ? _In2 : _In1); };
	/*maximal comparison*/
	inline int max(int _In1, int _In2) { return (_In1 <= _In2 ? _In2 : _In1); };

	/*Implement of round*/
	inline double round(double _In);

	//***Array Calculation***//

	/*
	turn the value of the input array into double type
	@_InMat: input mat(for opencv)
	@_flag: debug flag, return 1 when error
	*/
	inline double* array2double_3D(cv::Mat &_InMat, bool &_flag);		//if flag = 1 when error

	/*
	scaling the array, return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_scale: scale factor
	*/
	inline int arrayScaling_3D(double* _InArr, const int* dims, double _scale);

	/*
	find the maximum element in 3D array, return 2 when can't find the maximum (maybe zero or not exist),
	return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Max: the output maximum
	@_sign: positive value or negative value. 0: will find the maximum of all the positive elements;
	1: will find the absolute maximum of all the negative elements
	*/
	inline int findMax_3D(double* _InArr, const int* _dims, double &_Max, bool _sign = 0);

	/*
	find the maximum element in 2D array, return 2 when can't find the maximum (maybe zero or not exist)
	return 1 when error, otherwise 0
	@_InArr: input 2D array
	@_dims: the dimensions of the input array, should be width*height or cols*rows
	@_Max: the output maximum
	@_sign: positive value or negative value. 0: will find the maximum of all the positive elements;
	1: will find the absolute maximum of all the negative elements
	*/
	inline int findMax_2D(double* _InArr, const int* _dims, double &_Max, bool _sign = 0);

	/*
	find the minimum element in 3D array, return 2 when can't find the minimum
	return 1 when error, otherwise 0
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Max: the output minimum
	@_sign: positive value or negative value. 0: will find the minimum of all the positive elements;
	1: will find the absolute minimum of all the negative elements **(including zero)
	*/
	inline int findMin_3D(double* _InArr, const int* _dims, double &_Min, bool _sign = 0);

	/*
	find the minimum element in 2D array, return 2 when can't find the minimum
	return 1 when error, otherwise 0
	@_InArr: input 2D array
	@_dims: the dimensions of the input array, should be width*height or cols*rows
	@_Max: the output minimum
	@_sign: positive value or negative value. 0: will find the minimum of all the positive elements;
	1: will find the absolute minimum of all the negative elements **(including zero)
	*/
	inline int findMin_2D(double* _InArr, const int* _dims, double &_Min, bool _sign = 0);

	/*
	find the value range of the input 3D array
	@_InArr: input 3D array
	@_dims: the dimensions of the input array, should be width*height*depth or cols*rows*channels
	@_Range: the output range, will be (min,max)
	@_sign: whether signed value or not. 0: will find the value regardless of sign, that is, find the
	absolute value range; 1: will find the value considering the sign
	*/
	inline int findRange_3D(double* _InArr, const int* _dims, double* _Range, bool _sign = 0);

	/*
	find the value range of the input 2D array
	@_InArr: input 2D array
	@_dims: the dimensions of the input array, should be width*height or cols*rows
	@_Range: the output range, will be (min,max)
	@_sign: whether signed value or not. 0: will find the value regardless of sign, that is, find the
	absolute value range; 1: will find the value considering the sign
	*/
	inline int findRange_2D(double* _InArr, const int* _dims, double* _Range, bool _sign = 0);

	private:				//function for HOG

		/*make a HOG picture by HOG features*/
		int HOGpicture(mMat_D_3D _feat, int _GlyphSize, bool _load, cv::Mat &_OutPic);

		/*load the glyph model for default*/
		uchar* loadGlyph(std::string _filename, int _width, int _height, int _depth);

		/*load file*/
		std::vector<std::string> &split(const std::string &_s, char _delim, std::vector<std::string> &_elems);
		std::vector<std::string> split(const std::string &_s, char _delim);


		//可以做一樣的(input arguement不一樣)
		/*if can't find maximum , will return -1*/
		double findMaxElement3D(mMat_D_3D _mMat, bool _sign);
		double findMaxElement2D(cv::Mat _cvMat, bool _sign);
		double findMinElement2D(cv::Mat _cvMat);

		/*
		limit the range of matrix, return 1 when error, otherwise 0
		@_InOutMat: the input mat, and will return the result mat. **input mat type should be CV_64FC1
		@_upperBound: the upper boundary
		@_lowerBound: the lower boundary
		*/
		int cropValueRange(cv::Mat& _InOutMat, double _upperBound, double _lowerBound);			//just for 2D mat

		/*
		turn the value range to gray scale map
		http://www.programming-techniques.com/2013/01/contrast-stretching-using-c-and-opencv.html
		*/
		int grayscaling(cv::Mat& _InMat, cv::Mat& _OutMat, double _ValueUBound, double _ValueLBound);

		int GrayScaleCodec(double _In, double _ValueUBound, double _ValueLBound);

		/*
		turn gradient to flow field, will return the corresponding flow vector number of the input gradient vector
		number
		@_InOriNumber: the input gradient vector number
		*/
		double* Gradient2FFCodec(int _InOriNumber);

		/*
		*/
		double* GradientCodec(int _InOriNumber);

		int MatScaling2D(cv::Mat &_cvMat, double _scale);
		int MatScaling3D(cv::Mat &_cvMat, double _scale);

		/*
		print all the value in the 2D array
		@_InMat: the input array **the mat type should be CV_64FC1
		*/
		int printValues2D(cv::Mat _InMat);

public:
	fHOG();
	~fHOG();

	/*
	Calulate the HOG feature, return 1 when error, otherwise 0
	@_SrcImg: the input image
	@_CellSize: the cell size of HOG feature
	@_OutFeature: the output feature, should be GradientApp::mMat_D_3D
	@_Normalize: whether use nightbor normalization
	@_truncator: the upper bound of the feature value
	*/
	int extract_feature(cv::Mat& _SrcImg, int _CellSize, mMat_D_3D &_OutFeature, bool _Normalize = 1, double _truncator = 0.2);

	/*
	visualize the HOG feature like flow field, return 1 when error, otherwise 0
	@_HOGFeature: the input HOG Feature, should be GradientApp::mMat_D_3D struct, work with GradientApp::HOGFeature
	@_saveDir: the directory of the result image to save. If you want save the result, please
	enter the whole directory name, like: "DIR/FILENAME.DATATYPE". The default is not to save.
	**we won't create the directory which you want to save if it not exist
	*/
	int feature_visualize(mMat_D_3D _HOGFeature, std::string _saveDir = "");

};

