#include "fHOG.h"

double const fHOG::uu[9] = { 1.0000, 0.9397, 0.7660, 0.500, 0.1736,
-0.1736, -0.5000, -0.7660, -0.9397 };
double const fHOG::vv[9] = { 0.0000, 0.3420, 0.6428, 0.8660, 0.9848,
0.9848, 0.8660, 0.6428, 0.3420 };
const double fHOG::EPS = 0.0001;

/*the angle between the gradient and flow will be 90 degree */
double const fHOG::uu_f[9] = { 0.0000, 0.3420, 0.6428, 0.8660, 0.9848,
0.9848, 0.8660, 0.6428, 0.3420 };
double const fHOG::vv_f[9] = { 1.0000, 0.9397, 0.7660, 0.5000, 0.1736,
-0.1736, -0.5000, -0.7660, -0.9397 };

double fHOG::round(double _In)
{
	return static_cast<int>(_In + 0.5);
}

double* fHOG::array2double_3D(cv::Mat &_InMat, bool &_flag)
{
	if (_InMat.empty())
	{
		std::cout << "GradientApp::array2double_3D: input mat is empty" << std::endl;
		_flag = 1;
	}

	if (_InMat.type() == 16)		//8UC3
	{
		double *_OutArray = (double *)malloc(_InMat.rows* _InMat.cols * 3 * sizeof(double));
		uchar* ptr = _InMat.ptr<uchar>(0);

		for (size_t i = 0; i < _InMat.rows; i++)
		{
			for (size_t j = 0; j < _InMat.cols; j++)
			{
				*(_OutArray + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j);											//B level 0
				*(_OutArray + _InMat.cols * _InMat.rows + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j + 1);			//G level 1
				*(_OutArray + 2 * _InMat.cols * _InMat.rows + i * _InMat.cols + j) = (double)*(ptr + i * _InMat.cols * 3 + 3 * j + 2);		//R level 2
			}
		}
		_flag = 0;
		return _OutArray;
	}
	else
	{
		std::cout << "GradientApp::array2double_3D: input mat is not support" << std::endl;
		_flag = 1;
		double* unknowm = (double*)malloc(sizeof(double));
		return unknowm;
	}
}

int fHOG::arrayScaling_3D(double* _InArr, const int* dims, double _scale)
{
	if ((dims[0] == 0) || (dims[1] == 0) || (dims[2] == 0))
	{
		std::cout << "GradientApp::arrayScaling_3D: error dims in 3D" << std::endl;
		return 1;
	}
	if (_scale == 0.0)
	{
		std::cout << "GradientApp::arrayScaling_3D: scale can't be zero" << std::endl;
		return 1;
	}

	for (size_t i = 0; i < dims[1]; i++)
	{
		for (size_t j = 0; j < dims[0]; j++)
		{
			*(_InArr + i*dims[0] + j) = *(_InArr + i*dims[0] + j) / _scale;													//B
			*(_InArr + dims[0] * dims[1] + i*dims[0] + j) = *(_InArr + dims[0] * dims[1] + i*dims[0] + j) / _scale;			//G
			*(_InArr + 2 * dims[0] * dims[1] + i*dims[0] + j) = *(_InArr + 2 * dims[0] * dims[1] + i*dims[0] + j) / _scale;	//R
		}
	}
	return 0;
}

int fHOG::findMax_3D(double* _InArr, const int* _dims, double &_Max, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findMax_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double maximum = -1000000;
	if (!_sign)
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					maximum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) > maximum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) : maximum;
				}
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	else
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					maximum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) > maximum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) : maximum;
				}
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	return 0;
}

int fHOG::findMax_2D(double* _InArr, const int* _dims, double &_Max, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0))
	{
		std::cout << "GradientApp::findMax_3D: error dims in 2D" << std::endl;
		return 1;
	}
	double maximum = -1000000;
	if (!_sign)
	{
		for (size_t i = 0; i < _dims[1]; i++)
		{
			for (size_t j = 0; j < _dims[0]; j++)
			{
				maximum = *(_InArr + i*_dims[0] + j) > maximum ? *(_InArr + i*_dims[0] + j) : maximum;
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	else
	{
		for (size_t i = 0; i < _dims[1]; i++)
		{
			for (size_t j = 0; j < _dims[0]; j++)
			{
				maximum = *(_InArr + i*_dims[0] + j)*(-1) > maximum ? *(_InArr + i*_dims[0] + j)*(-1) : maximum;
			}
		}
		_Max = maximum;
		if (maximum <= 0)
		{
			return 2;
		}
	}
	return 0;
}

int fHOG::findMin_3D(double* _InArr, const int* _dims, double &_Min, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findMin_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double minimum = 1000000;
	if (!_sign)
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					if (*(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) >= 0)
						minimum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) < minimum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j) : minimum;

				}
			}
		}
		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	else
	{
		for (size_t k = 0; k < _dims[2]; k++)
		{
			for (size_t i = 0; i < _dims[1]; i++)
			{
				for (size_t j = 0; j < _dims[0]; j++)
				{
					if (*(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) >= 0)
						minimum = *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) < minimum ? *(_InArr + k*_dims[0] * _dims[1] + i*_dims[0] + j)*(-1) : minimum;
				}
			}
		}
		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	return 0;
}

int fHOG::findMin_2D(double* _InArr, const int* _dims, double &_Min, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0))
	{
		std::cout << "GradientApp::findMin_2D: error dims in 2D" << std::endl;
		return 1;
	}

	double minimum = 1000000;
	if (!_sign)
	{

		for (size_t i = 0; i < _dims[1]; i++)
		{
			for (size_t j = 0; j < _dims[0]; j++)
			{
				if (*(_InArr + i*_dims[0] + j) >= 0)
					minimum = *(_InArr + i*_dims[0] + j) < minimum ? *(_InArr + i*_dims[0] + j) : minimum;

			}
		}

		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	else
	{
		for (size_t i = 0; i < _dims[1]; i++)
		{
			for (size_t j = 0; j < _dims[0]; j++)
			{
				if (*(_InArr + i*_dims[0] + j)*(-1) >= 0)
					minimum = *(_InArr + i*_dims[0] + j)*(-1) < minimum ? *(_InArr + i*_dims[0] + j)*(-1) : minimum;
			}
		}

		_Min = minimum;
		if (minimum == 1000000)
		{
			return 2;
		}
	}
	return 0;
}

int fHOG::findRange_3D(double* _InArr, const int* _dims, double* _Range, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0) || (_dims[2] == 0))
	{
		std::cout << "GradientApp::findRange_3D: error dims in 3D" << std::endl;
		return 1;
	}

	double max_s, max_us, min_s, min_us;
	int Max1 = findMax_3D(_InArr, _dims, max_us, 0);		//positive maximum
	int Max2 = findMax_3D(_InArr, _dims, max_s, 1);		//negative maximum
	int Min1 = findMin_3D(_InArr, _dims, min_us, 0);		//positive minimum(include 0)
	int Min2 = findMin_3D(_InArr, _dims, min_s, 1);		//negative minimum(include 0)

	if (!_sign)
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_3D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if ((Max1 == 2) && (Max2 == 2))
			*(_Range + 1) = 0.0;
		else if ((Max1 == 0) && (Max2 == 2))
			*(_Range + 1) = max_us;
		else if ((Max1 == 2) && (Max2 == 0))
			*(_Range + 1) = max_s;
		else if ((Max1 == 0) && (Max2 == 0))
			*(_Range + 1) = max(max_us, max_s);

		/*Minimum*/
		if ((Min1 == 2) && (Min2 == 2))
		{
			std::cout << "GradientApp::findRange_3D: Something ERROR" << std::endl;
			return 1;
		}
		else if ((Min1 == 0) && (Min2 == 2))
			*_Range = min_us;
		else if ((Min1 == 2) && (Min2 == 0))
			*_Range = min_s;
		else if ((Min1 == 0) && (Min2 == 0))
			*_Range = min(min_us, min_s);
	}
	else
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_3D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if (Max1 == 0)
			*(_Range + 1) = max_us;
		else           //no positive elements
			*(_Range + 1) = min_s;


		/*Minimum*/
		if (Max2 == 0)
			*_Range = max_s;
		else           //no negative elements
			*_Range = min_us;
	}
	return 0;
}

int fHOG::findRange_2D(double* _InArr, const int* _dims, double* _Range, bool _sign /*= 0*/)
{
	if ((_dims[0] == 0) || (_dims[1] == 0))
	{
		std::cout << "GradientApp::findRange_2D: error dims in 2D" << std::endl;
		return 1;
	}

	double max_s, max_us, min_s, min_us;
	int Max1 = findMax_2D(_InArr, _dims, max_us, 0);		//positive maximum
	int Max2 = findMax_2D(_InArr, _dims, max_s, 1);			//negative maximum
	int Min1 = findMin_2D(_InArr, _dims, min_us, 0);		//positive minimum(include 0)
	int Min2 = findMin_2D(_InArr, _dims, min_s, 1);			//negative minimum(include 0)

	if (!_sign)
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_2D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if ((Max1 == 2) && (Max2 == 2))
			*(_Range + 1) = 0.0;
		else if ((Max1 == 0) && (Max2 == 2))
			*(_Range + 1) = max_us;
		else if ((Max1 == 2) && (Max2 == 0))
			*(_Range + 1) = max_s;
		else if ((Max1 == 0) && (Max2 == 0))
			*(_Range + 1) = max(max_us, max_s);

		/*Minimum*/
		if ((Min1 == 2) && (Min2 == 2))
		{
			std::cout << "GradientApp::findRange_2D: Something ERROR" << std::endl;
			return 1;
		}
		else if ((Min1 == 0) && (Min2 == 2))
			*_Range = min_us;
		else if ((Min1 == 2) && (Min2 == 0))
			*_Range = min_s;
		else if ((Min1 == 0) && (Min2 == 0))
			*_Range = min(min_us, min_s);
	}
	else
	{
		if ((Max1 == 1) || (Max2 == 1) || (Min1 == 1) || (Min2 == 1))
		{
			std::cout << "GradientApp::findRange_2D: ERROR" << std::endl;
			return 1;
		}

		/*Maximum*/
		if (Max1 == 0)
			*(_Range + 1) = max_us;
		else           //no positive elements
			*(_Range + 1) = min_s;


		/*Minimum*/
		if (Max2 == 0)
			*_Range = max_s;
		else           //no negative elements
			*_Range = min_us;
	}
	return 0;
}

int fHOG::HOGpicture(mMat_D_3D _feat, int _GlyphSize, bool _load, cv::Mat &_OutPic)
{
	uchar* bim;
	if (!_load)
	{
		cv::Mat bim1 = cv::Mat::zeros(_GlyphSize, _GlyphSize, CV_8UC1);
		uchar* rows = bim1.ptr<uchar>(0);
		for (size_t i = 0; i < 20; i++)
		{
			*(rows + i * 20 + 10) = 1;
			*(rows + i * 20 + 11) = 1;
		}
		bim = (uchar*)malloc(_GlyphSize * _GlyphSize * 9 * sizeof(uchar));
		for (size_t i = 0; i < _GlyphSize*_GlyphSize; i++)
		{
			*(bim + i) = *(rows + i);
		}

		for (int i = 1; i < 9; i++)
		{
			cv::Mat rotationMatA = getRotationMatrix2D(cv::Point(10.5, 10.5), -i * 20, 1.0);
			cv::Mat rotA;
			cv::warpAffine(bim1, rotA, rotationMatA, bim1.size());
			uchar* rowsA = rotA.ptr<uchar>(0);

			for (size_t j = 0; j < _GlyphSize * _GlyphSize; j++)
			{
				*(bim + i*_GlyphSize*_GlyphSize + j) = *rowsA++;
			}
		}
	}
	else
		bim = loadGlyph("Glyph_20.txt", 20, 20, 9);

	/* make pictures of positive weights bs adding up weighted glyphs*/
	for (size_t i = 0; i < _feat.dims0 * _feat.dims1 * _feat.dims2; i++)
	{
		*(_feat.data + i) = *(_feat.data + i) < 0 ? 0 : *(_feat.data + i);
	}

	cv::Mat out = cv::Mat::zeros(_feat.dims1*_GlyphSize, _feat.dims0*_GlyphSize, CV_64FC1);
	double* ptr = out.ptr<double>(0);
	for (size_t i = 0; i < _feat.dims1; i++)
	{
		int iis[2] = { i*_GlyphSize, (i + 1)*_GlyphSize };
		for (size_t j = 0; j < _feat.dims0; j++)
		{
			int jjs[2] = { j*_GlyphSize, (j + 1)*_GlyphSize };
			for (size_t k = 0; k < 9; k++)
			{
				for (size_t l = iis[0]; l < iis[1]; l++)
				{
					for (size_t m = jjs[0]; m < jjs[1]; m++)
					{
						*(ptr + l*_feat.dims0*_GlyphSize + m) = *(ptr + l*_feat.dims0*_GlyphSize + m) + *(bim + k*_GlyphSize*_GlyphSize + (l % 20)*_GlyphSize + (m % 20))*
							*(_feat.data + (k + 18)*_feat.dims0*_feat.dims1 + i*_feat.dims0 + j);
					}
				}
			}
		}
	}

	free(bim);

	int dims[3];
	dims[0] = _feat.dims0;
	dims[1] = _feat.dims1;
	dims[2] = _feat.dims2;
	double* range = (double*)malloc(2 * sizeof(double));
	int flag = findRange_3D(_feat.data, dims, range);
	if (flag == 1)
	{
		std::cout << "GradientApp::HOGpicture: ERROR" << std::endl;
		return 1;
	}

	MatScaling2D(out, range[1]);

	out.assignTo(_OutPic);
	return 0;
}

uchar* fHOG::loadGlyph(std::string _filename, int _width, int _height, int _depth)
{
	std::ifstream fin(_filename);

	uchar* bim = (uchar*)malloc(_height * _width * _depth * sizeof(uchar));
	uchar* bimTemp = bim;
	std::string lineTemp;
	int count = 0;
	while (std::getline(fin, lineTemp))
	{
		count++;
		if ((count % (_height + 1)) == 0)
			continue;
		std::vector<std::string> lineElems = split(lineTemp, '\t');
		if (lineElems.size() != _width)
		{
			std::cout << "GradientApp::loadGlyph: data structure of the input file is error" << std::endl;
			break;
		}
		for (size_t i = 0; i < _width; i++)
		{
			*bimTemp++ = (uchar)std::stoi(lineElems.at(i));
		}
	}
	return bim;
}

std::vector<std::string> & fHOG::split(const std::string &_s, char _delim, std::vector<std::string> &_elems)
{
	std::stringstream ss(_s);
	std::string item;
	while (std::getline(ss, item, _delim))
		_elems.push_back(item);
	return _elems;
}

std::vector<std::string> fHOG::split(const std::string &_s, char _delim)
{
	std::vector<std::string> _elems;
	split(_s, _delim, _elems);
	return _elems;
}

double fHOG::findMaxElement3D(mMat_D_3D _mMat, bool _sign)
{
	double maximum = -100;
	if (!_sign)
	{
		for (size_t k = 0; k < _mMat.dims2; k++)
		{
			for (size_t i = 0; i < _mMat.dims1; i++)
			{
				for (size_t j = 0; j < _mMat.dims0; j++)
				{
					maximum = *(_mMat.data + k*_mMat.dims0*_mMat.dims1 + i*_mMat.dims0 + j) > maximum ? *(_mMat.data + k*_mMat.dims0*_mMat.dims1 + i*_mMat.dims0 + j) : maximum;
				}
			}
		}
		if (maximum <= 0)
		{
			maximum = 1;
		}
	}
	else
	{
		for (size_t k = 0; k < _mMat.dims2; k++)
		{
			for (size_t i = 0; i < _mMat.dims1; i++)
			{
				for (size_t j = 0; j < _mMat.dims0; j++)
				{
					maximum = *(_mMat.data + k*_mMat.dims0*_mMat.dims1 + i*_mMat.dims0 + j)*(-1) > maximum ? *(_mMat.data + k*_mMat.dims0*_mMat.dims1 + i*_mMat.dims0 + j)*(-1) : maximum;
				}
			}
		}
		if (maximum <= 0)
		{
			maximum = 1;
		}
	}

	return maximum;
}

double fHOG::findMaxElement2D(cv::Mat _cvMat, bool _sign)
{
	double* ptr = _cvMat.ptr<double>(0);
	double maximum = -100;
	if (!_sign)
	{

		for (size_t i = 0; i < _cvMat.rows; i++)
		{
			for (size_t j = 0; j < _cvMat.cols; j++)
			{
				maximum = *(ptr + i*_cvMat.cols + j) > maximum ? *(ptr + i*_cvMat.cols + j) : maximum;
			}
		}

		if (maximum < 0)
		{
			maximum = -1;
		}
	}
	else
	{
		for (size_t i = 0; i < _cvMat.rows; i++)
		{
			for (size_t j = 0; j < _cvMat.cols; j++)
			{
				maximum = *(ptr + i*_cvMat.cols + j)*(-1) > maximum ? *(ptr + i*_cvMat.cols + j)*(-1) : maximum;
			}
		}
		if (maximum < 0)
		{
			maximum = -1;
		}
	}

	return maximum;
}

double fHOG::findMinElement2D(cv::Mat _cvMat)
{
	double* ptr = _cvMat.ptr<double>(0);
	double minimum = 100;

	for (size_t i = 0; i < _cvMat.rows; i++)
	{
		for (size_t j = 0; j < _cvMat.cols; j++)
		{
			minimum = *(ptr + i*_cvMat.cols + j) < minimum ? *(ptr + i*_cvMat.cols + j) : minimum;
		}
	}

	return minimum;
}

int fHOG::cropValueRange(cv::Mat& _InOutMat, double _upperBound, double _lowerBound)
{
	if (_InOutMat.empty())
	{
		std::cout << "GradientApp::cropValueRange: input is empty" << std::endl;
		return 1;
	}
	if (_lowerBound >= _upperBound)
	{
		std::cout << "GradientApp::cropValueRange: upper bound must be larger than lower bound" << std::endl;
		return 1;
	}
	if (_InOutMat.type() != 6)
	{
		std::cout << "GradientApp::cropValueRange: input mat type should be CV_64FC1" << std::endl;
		return 1;
	}
	double* ptr = _InOutMat.ptr<double>(0);
	for (size_t i = 0; i < _InOutMat.rows; i++)
	{
		for (size_t j = 0; j < _InOutMat.cols; j++)
		{
			*(ptr + i*_InOutMat.cols + j) = *(ptr + i*_InOutMat.cols + j) < _lowerBound ? _lowerBound : *(ptr + i*_InOutMat.cols + j);
			*(ptr + i*_InOutMat.cols + j) = *(ptr + i*_InOutMat.cols + j) > _upperBound ? _upperBound : *(ptr + i*_InOutMat.cols + j);
		}
	}

	return 0;
}

int fHOG::grayscaling(cv::Mat& _InMat, cv::Mat& _OutMat, double _ValueUBound, double _ValueLBound)
{
	if (_InMat.empty())
	{
		std::cout << "GradientApp::grayscaling: input is empty" << std::endl;
		system("pause");
		return 1;
	}
	if (_ValueLBound >= _ValueUBound)
	{
		std::cout << "GradientApp::grayscaling: upper bound must be larger than lower bound" << std::endl;
		system("pause");
		return 1;
	}

	_OutMat = cv::Mat(_InMat.rows, _InMat.cols, CV_8UC1);

	double* ptrSRC = _InMat.ptr<double>(0);
	uchar* ptrDST = _OutMat.ptr<uchar>(0);
	for (size_t i = 0; i < _OutMat.rows; i++)
	{
		for (size_t j = 0; j < _OutMat.cols; j++)
		{
			*(ptrDST + i*_OutMat.cols + j) = (uchar)GrayScaleCodec(*(ptrSRC + i*_InMat.cols + j), _ValueUBound, _ValueLBound);
		}
	}
	return 0;
}

int fHOG::GrayScaleCodec(double _In, double _ValueUBound, double _ValueLBound)
{
	if (_ValueUBound <= _ValueLBound)
	{
		std::cout << "GradientApp::GrayScaleCodec: upper bound must be larger than lower bound" << std::endl;
		system("pause");
		return -1;
	}
	double m = (255 - 0) / (_ValueUBound - _ValueLBound);

	return m*(_In - _ValueLBound);
}

double* fHOG::Gradient2FFCodec(int _InOriNumber)
{
	double* out = (double*)malloc(2 * sizeof(double));
	int sign_v[2] = { 1, -1 };
	int sign = _InOriNumber / 9;
	int num = _InOriNumber % 9;
	if (sign > 1)
	{
		std::cout << "GradientApp::Gradient2FFCodec: please enter 0~17, or it will return valid value" << std::endl;
		return out;
	}
	out[0] = uu_f[num] * sign_v[sign];
	out[1] = vv_f[num] * sign_v[sign];

	return out;
}

double* fHOG::GradientCodec(int _InOriNumber)
{
	double* out = (double*)malloc(2 * sizeof(double));
	int sign_v[2] = { 1, -1 };
	int sign = _InOriNumber / 9;
	int num = _InOriNumber % 9;
	if (sign > 1)
	{
		std::cout << "GradientApp::Gradient2FFCodec: please enter 0~17, or it will return valid value" << std::endl;
		return out;
	}
	out[0] = uu[num] * sign_v[sign];
	out[1] = vv[num] * sign_v[sign];

	return out;
}

int fHOG::MatScaling2D(cv::Mat &_cvMat, double _scale)
{
	if (_scale == 0)
	{
		std::cout << "GradientApp::MatScaling: Denominator should not be zero" << std::endl;
		system("pause");
		return 1;
	}

	double* ptr = _cvMat.ptr<double>(0);
	for (size_t i = 0; i < _cvMat.rows; i++)
	{
		for (size_t j = 0; j < _cvMat.cols; j++)
		{
			*(ptr + i*_cvMat.cols + j) = *(ptr + i*_cvMat.cols + j) / _scale;
		}
	}

	return 0;
}

int fHOG::MatScaling3D(cv::Mat &_cvMat, double _scale)
{
	if (_scale == 0)
	{
		std::cout << "GradientApp::MatScaling: Denominator should not be zero" << std::endl;
		system("pause");
		return 1;
	}

	if (_cvMat.type() == 3)
	{
		double* ptr = _cvMat.ptr<double>(0);
		for (size_t i = 0; i < _cvMat.rows; i++)
		{
			for (size_t j = 0; j < _cvMat.cols; j++)
			{
				*(ptr + i*_cvMat.cols * 3 + 3 * j) = *(ptr + i*_cvMat.cols * 3 + 3 * j) / _scale;			//B
				*(ptr + i*_cvMat.cols * 3 + 3 * j + 1) = *(ptr + i*_cvMat.cols * 3 + 3 * j + 1) / _scale;	//G
				*(ptr + i*_cvMat.cols * 3 + 3 * j + 2) = *(ptr + i*_cvMat.cols * 3 + 3 * j + 2) / _scale;	//R
			}
		}
	}
	else
	{
		std::cout << "GradientApp::MatScaling: input mat type are not supported" << std::endl;
		system("pause");
		return 1;
	}
	return 0;
}

int fHOG::printValues2D(cv::Mat _InMat)
{
	if (_InMat.empty() == 1)
	{
		std::cout << "GradientApp::printValues: the input mat is empty" << std::endl;
		return 1;
	}
	if (_InMat.type() != 6)
	{
		std::cout << "GradientApp::printValues: the input mat type should be CV_64FC1" << std::endl;
		return 1;
	}

	double* ptr = _InMat.ptr<double>(0);
	std::vector<double> set;
	int init = 0;
	for (size_t i = 0; i < _InMat.rows; i++)
	{
		for (size_t j = 0; j < _InMat.cols; j++)
		{
			if (init == 0)
			{
				init = 1;
				set.push_back(*(ptr + i*_InMat.cols + j));
				continue;
			}

			int flag = 0;
			for (size_t k = 0; k < set.size(); k++)
			{
				if (*(ptr + i*_InMat.cols + j) == *(set.data() + k))
				{
					flag = 1;
					break;
				}
			}
			if (flag == 0) set.push_back(*(ptr + i*_InMat.cols + j));
		}
	}

	if (set.size() > 1000)
	{
		std::cout << "The number of value in the 2D array is too large." << std::endl;
		std::cout << "Are you want to it all? (y/n)" << "  ";
		char in;
		std::cin >> in;
		if ((in == 'y') || (in == 'Y'))
		{
			double* temp = set.data();
			int count = 0;
			for (size_t i = 0; i < set.size(); i++)
			{
				count++;
				std::cout << *(temp + i) << "  ";
				if (count % 8 == 0) std::cout << std::endl;
			}
		}
		else
		{
			std::cout << "..." << std::endl;
		}
	}
	else
	{
		double* temp = set.data();
		int count = 0;
		for (size_t i = 0; i < set.size(); i++)
		{
			count++;
			std::cout << *(temp + i) << "  ";
			if (count % 8 == 0) std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	std::cout << "Total: " << set.size() << std::endl;

	return 0;
}

fHOG::fHOG()
{
}


fHOG::~fHOG()
{
}

int fHOG::extract_feature(cv::Mat& _SrcImg, int _CellSize, mMat_D_3D &_OutFeature, bool _Normalize /*= 1*/, double _truncator /*= 0.2*/)
{
	if (_SrcImg.empty())
	{
		std::cout << "GradientApp::HOGFeature: input image is empty" << std::endl;
		system("pause");
		return 1;
	}
	const int *dims = _SrcImg.size.p;		//dims[0] = rows/height, dims[1] = cols/width

	bool flag;
	double *SrcImg = array2double_3D(_SrcImg, flag);
	if (flag == 1)
	{
		std::cout << "GradientApp::HOGFeature: ERROR" << std::endl;
		return 1;
	}

	double *range = (double*)malloc(2 * sizeof(double));
	int dimTemp[3];
	dimTemp[0] = dims[1];
	dimTemp[1] = dims[0];
	dimTemp[2] = 3;

	if (findRange_3D(SrcImg, dimTemp, range, 0))
	{
		std::cout << "GradientApp::HOGFeature: ERROR" << std::endl;
		return 1;
	}

	if (arrayScaling_3D(SrcImg, dimTemp, 255))
	{
		std::cout << "GradientApp::HOGFeature: ERROR" << std::endl;
		return 1;
	}

	/*memory for caching orientation histograms & their norms*/
	int blocks[2];		//cell size
	blocks[0] = (int)round((double)dims[1] / (double)_CellSize);				//cell size in cols
	blocks[1] = (int)round((double)dims[0] / (double)_CellSize);				//cell size in rows

	double *hist = (double *)malloc(blocks[0] * blocks[1] * 18 * sizeof(double));
	double *norm = (double *)malloc(blocks[0] * blocks[1] * sizeof(double));

	/*memory for HOG features*/
	int out[3];		//dims of output
	out[0] = max(blocks[0] - 2, 0);
	out[1] = max(blocks[1] - 2, 0);
	out[2] = 27 + 4 + 1;
	_OutFeature.data = (double *)malloc(out[0] * out[1] * out[2] * sizeof(double));
	_OutFeature.dims0 = out[0];
	_OutFeature.dims1 = out[1];
	_OutFeature.dims2 = out[2];

	int visible[2];									//range of selected pixel for visualization
	visible[0] = blocks[0] * _CellSize;
	visible[1] = blocks[1] * _CellSize;

	for (int x = 1; x < visible[1] - 1; x++) {
		for (int y = 1; y < visible[0] - 1; y++) {

			//-----------------illustration-------------------//
			//		x x x x x x x x x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x o o o o o o o x
			//		x x x x x x x x x
			//		o: pixel which to cal
			//		x: pixel which not to cal (margin pixel)
			//------------------------------------------------//

			/* first color channel: B*/
			double *s = SrcImg + min(x, dims[0] - 2)*dims[1] + min(y, dims[1] - 2);		//x bound: dims[1] - 2, y bound: dims[0] - 2
			double dx3 = *(s + 1) - *(s - 1);			//delta cols
			double dy3 = *(s + dims[1]) - *(s - dims[1]);		//delta rows
			double v3 = dx3*dx3 + dy3*dy3;

			/* second color channel: G*/
			s += dims[0] * dims[1];
			double dx2 = *(s + 1) - *(s - 1);
			double dy2 = *(s + dims[1]) - *(s - dims[1]);
			double v2 = dx2*dx2 + dy2*dy2;

			/* third color channel: R*/
			s += dims[0] * dims[1];
			double dx = *(s + 1) - *(s - 1);
			double dy = *(s + dims[1]) - *(s - dims[1]);
			double v = dx*dx + dy*dy;

			/* pick channel with strongest gradient*/
			if (v2 > v) {
				v = v2;
				dx = dx2;
				dy = dy2;
			}
			if (v3 > v) {
				v = v3;
				dx = dx3;
				dy = dy3;
			}

			/* snap to one of 18 orientations, use vector projection*/
			double best_dot = 0;
			int best_o = 0;
			for (int o = 0; o < 9; o++) {
				double dot = uu[o] * dx + vv[o] * dy;

				if (dot > best_dot) {				//positive orientation
					best_dot = dot;
					best_o = o;

				}
				else if (-dot > best_dot) {			//negative orientation
					best_dot = -dot;
					best_o = o + 9;

				}
			}

			/* add to 4 histograms around pixel using linear interpolation, calculate the pixel gradient contribution to neighbor
			four cell*/
			double xp = ((double)x + 0.5) / (double)_CellSize - 0.5;
			double yp = ((double)y + 0.5) / (double)_CellSize - 0.5;
			int ixp = (int)floor(xp);
			int iyp = (int)floor(yp);
			double vx0 = xp - ixp;
			double vy0 = yp - iyp;
			double vx1 = 1.0 - vx0;
			double vy1 = 1.0 - vy0;
			v = sqrt(v);

			//-------------illustration--------------------------------//
			//	hist + ixp*blocks[0] + iyp -> cell central position
			//  (3*3 for example)
			//	 _____________________________            _____________________________			  _____________________________
			//	|         |         |         |	         |    |    |    |    |    |    |		 |    |         |         |    |
			//	|         |         |         |	         |----.----|----.----|----.----|			 |----.---------.---------.----|
			//	|_________|_________|_________|	         |____|____|____|____|____|____|		 |    |         |         |    |
			//	|         |         |         |	         |    |    |    |    |    |    |		 |    |         |         |    |
			//	|         |         |         |	  ===>   |----.----|----.----|----.----|  ===>	 |----.---------.---------.----|
			//	|_________|_________|_________|	         |____|____|____|____|____|____|		 |    |         |         |    |
			//	|         |         |         |	         |    |    |    |    |    |    |		 |    |         |         |    |
			//	|         |         |         |	         |----.----|----.----|----.----|			 |----.---------.---------.----|
			//	|_________|_________|_________|	         |____|____|____|____|____|____|		 |____|_________|_________|____|
			//
			//             hist(3*3)                            cell central crop                           (ixp,iyp) (4*4)
			//
			//	ixp, iyp range: -1 ~ (dims-1)
			//
			//	best_o*blocks[0]*blocks[1] -> orientation level
			//
			//		         4
			//		    ___________
			//		   /		  /|
			//		4 /          / |
			//		 /__________/  |  orientation level = 18  (4*4 for example)
			//		 |          |  /
			//		 |          | /
			//		 |__________|/
			//
			//	vx1*vy1*v -> Gaussian Weighting
			//---------------------------------------------------------//

			if (ixp >= 0 && iyp >= 0) {					//right-bottom part
				*(hist + ixp*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx1*vy1*v;
			}

			if (ixp + 1 < blocks[1] && iyp >= 0) {		//left-bottom part
				*(hist + (ixp + 1)*blocks[0] + iyp + best_o*blocks[0] * blocks[1]) +=
					vx0*vy1*v;
			}

			if (ixp >= 0 && iyp + 1 < blocks[0]) {		//right-top part
				*(hist + ixp*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx1*vy0*v;
			}

			if (ixp + 1 < blocks[1] && iyp + 1 < blocks[0]) {		//left-top part
				*(hist + (ixp + 1)*blocks[0] + (iyp + 1) + best_o*blocks[0] * blocks[1]) +=
					vx0*vy0*v;
			}
		}
	}

	/* compute energy in each block by summing over orientations*/
	for (int o = 0; o < 9; o++) {
		double *src1 = hist + o*blocks[0] * blocks[1];
		double *src2 = hist + (o + 9)*blocks[0] * blocks[1];
		double *dst = norm;
		double *end = norm + blocks[1] * blocks[0];
		while (dst < end) {
			*(dst++) += (*src1 + *src2) * (*src1 + *src2);
			src1++;
			src2++;
		}
	}

	/* compute features*/
	for (int x = 0; x < out[1]; x++) {
		for (int y = 0; y < out[0]; y++) {
			double *dst = _OutFeature.data + x*out[0] + y;
			double *src, *p, n1, n2, n3, n4;

			if (_Normalize) {
				p = norm + (x + 1)*blocks[0] + y + 1;
				n1 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + EPS);
				p = norm + (x + 1)*blocks[0] + y;
				n2 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + EPS);
				p = norm + x*blocks[0] + y + 1;
				n3 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + EPS);
				p = norm + x*blocks[0] + y;
				n4 = 1.0 / sqrt(*p + *(p + 1) + *(p + blocks[0]) + *(p + blocks[0] + 1) + EPS);
			}
			else  {
				n1 = 0.15;
				n2 = 0.15;
				n3 = 0.15;
				n4 = 0.15;
			}

			double t1 = 0;
			double t2 = 0;
			double t3 = 0;
			double t4 = 0;

			// contrast-sensitive features
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 18; o++) {
				double h1 = min(*src * n1, _truncator);
				double h2 = min(*src * n2, _truncator);
				double h3 = min(*src * n3, _truncator);
				double h4 = min(*src * n4, _truncator);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				t1 += h1;
				t2 += h2;
				t3 += h3;
				t4 += h4;
				dst += out[0] * out[1];
				src += blocks[0] * blocks[1];
			}

			// contrast-insensitive features
			src = hist + (x + 1)*blocks[0] + (y + 1);
			for (int o = 0; o < 9; o++) {
				double sum = *src + *(src + 9 * blocks[0] * blocks[1]);
				double h1 = min(sum * n1, _truncator);
				double h2 = min(sum * n2, _truncator);
				double h3 = min(sum * n3, _truncator);
				double h4 = min(sum * n4, _truncator);
				*dst = 0.5 * (h1 + h2 + h3 + h4);
				dst += out[0] * out[1];
				src += blocks[0] * blocks[1];
			}

			// texture features
			*dst = 0.2357 * t1;
			dst += out[0] * out[1];
			*dst = 0.2357 * t2;
			dst += out[0] * out[1];
			*dst = 0.2357 * t3;
			dst += out[0] * out[1];
			*dst = 0.2357 * t4;

			// truncation feature
			dst += out[0] * out[1];
			*dst = 0;
		}
	}

	free(hist);
	free(norm);
	free(SrcImg);
	return 0;
}

int fHOG::feature_visualize(mMat_D_3D _HOGFeature, std::string _saveDir /*= ""*/)
{
	mMat_D_3D HOGFeature(_HOGFeature.dims0, _HOGFeature.dims1, _HOGFeature.dims2, _HOGFeature.data);

	double* feat = (double*)malloc(HOGFeature.dims0*HOGFeature.dims1 * 9 * sizeof(double));
	for (size_t k = 0; k < 27; k++)
	{
		for (size_t i = 0; i < HOGFeature.dims1; i++)
		{
			for (size_t j = 0; j < HOGFeature.dims0; j++)
			{
				*(feat + (k % 9)*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) =
					*(feat + (k % 9)*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) +
					*(HOGFeature.data + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j);
			}
		}
	}

	/*calculate the average*/
	for (size_t k = 0; k < 9; k++)
	{
		for (size_t i = 0; i < HOGFeature.dims1; i++)
		{
			for (size_t j = 0; j < HOGFeature.dims0; j++)
			{
				*(feat + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) =
					*(feat + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) / 3;
			}
		}
	}

	/*remat the mat*/
	for (size_t k = 0; k < 9; k++)
	{
		for (size_t i = 0; i < HOGFeature.dims1; i++)
		{
			for (size_t j = 0; j < HOGFeature.dims0; j++)
			{
				*(HOGFeature.data + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) =
					*(feat + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j);
				*(HOGFeature.data + (k + 9)*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) =
					*(feat + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j);
				*(HOGFeature.data + (k + 18)*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) =
					*(feat + k*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j);
			}
		}
	}
	for (size_t k = 0; k < 5; k++)
	{
		for (size_t i = 0; i < HOGFeature.dims1; i++)
		{
			for (size_t j = 0; j < HOGFeature.dims0; j++)
			{
				*(HOGFeature.data + (k + 27)*HOGFeature.dims1*HOGFeature.dims0 + i*HOGFeature.dims0 + j) = 0;
			}
		}
	}

	cv::Mat show;
	HOGpicture(HOGFeature, 20, 1, show);

	if (cropValueRange(show, 1, 0))
	{
		std::cout << "GradientApp::ShowHOG: ERROR" << std::endl;
		system("pause");
		return 1;
	}

	int dims[2] = { show.cols, show.rows };
	double* range = (double*)malloc(2 * sizeof(double));
	double* ptr = show.ptr<double>(0);
	if (findRange_2D(ptr, dims, range))
	{
		std::cout << "GradientApp::ShowHOG: ERROR" << std::endl;
		return 1;
	}

	printValues2D(show);

	cv::Mat show_gray;
	if (grayscaling(show, show_gray, range[1], range[0]))
	{
		std::cout << "GradientApp::ShowHOG: ERROR" << std::endl;
		system("pause");
		return 1;
	}
	if (_saveDir != "")
	{
		cv::imwrite(_saveDir, show_gray);
	}

	cv::imshow("HOG Feature", show_gray);
	cv::waitKey(1);

	return 0;
}
