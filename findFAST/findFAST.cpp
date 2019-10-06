// findFAST.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2\highgui\highgui.hpp>//OpenCV图形处理头文件
#include <opencv2/opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

bool myFAST(Mat image, Point point);
void ORBdemo();
int *midFilter(int move[], int size);
int *sort(int *src, int size);
	

Mat preImages, images;
int Hession = 400;

int main()
{
	//----------------------------FAST特征点--------------------------
	Mat image;
	image = imread("timg.jpg",1);
	vector<KeyPoint> keyPoints;
	Mat imageG;
	cvtColor(image, imageG, COLOR_RGB2GRAY);
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(40);
	detector->detect(imageG, keyPoints);
	drawKeypoints(imageG,keyPoints, imageG, Scalar::all(-1), DrawMatchesFlags::DRAW_OVER_OUTIMG);

	namedWindow("test", WINDOW_AUTOSIZE);
	imshow("test", imageG);

	//----------------------------ORB特征匹配-------------------------
	VideoCapture cap("yTest.mp4");

	//判断文件存在
	if (!cap.isOpened())
		cout << "文件打开失败" << endl;

	//获取视频信息--总帧数
	long totalFrameNum = cap.get(CAP_PROP_FRAME_COUNT);
	cout << "总共的帧数为" << totalFrameNum << endl;

	long frameStrat = 5;

	//设置视频开始帧数
	cap.set(CAP_PROP_POS_FRAMES, frameStrat);

	int frameStop = 160;

	//获取视频信息--帧率
	double rate = cap.get(CAP_PROP_FPS);
	cout << "帧率为:" << rate << endl;

	namedWindow("ORBdemo", WINDOW_AUTOSIZE);

	int stop = true;
	int num = frameStrat;

	cap.read(preImages);
	cap.read(images);
	
	
	cout << "rows=" << images.rows << " cols=" << images.cols << endl;
	while (stop)
	{
		if (num > frameStrat)
			preImages = images.clone();

		//读取视频为图片
		if (!cap.read(images)) {
			cout << "读取视频失败" << endl;
			return -1;
		}

		if (num > frameStrat) {
			ORBdemo();

			if (num > frameStop - 1) {
				stop = false;
			}

			cout << "正在读取第" << num << "帧" << endl;
		}
		num++;
		waitKey(1);

	}
	cout << "over" << endl;
	waitKey(-1);
}

//orb特征匹配
void ORBdemo() {
	
	//操作系统启动到现在的时间（毫秒）
	double t1 = getTickCount();

	Mat rectImage = preImages.clone();
	rectImage = rectImage(Rect(preImages.cols/2 - 200, preImages.rows/2 - 100, 400, 200));
	//cout << rectImage.cols << " --rectImage-- " << rectImage.rows << endl;

	//特征点提取
	Ptr<ORB> detector = ORB::create(400);
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;

	//定义描述子
	Mat descriptor_obj, descriptor_scene;
	
	detector->detectAndCompute(preImages, Mat(), keypoints_obj, descriptor_obj);
	detector->detectAndCompute(images, Mat(), keypoints_scene, descriptor_scene);

	//计算 计算用的时间
	double t2 = getTickCount();
	double t = (t2 - t1) * 1000 / getTickFrequency();

	cout << "计算特征点花费时间" << t << endl;
	//特征匹配
	FlannBasedMatcher fbmatcher(new flann::LshIndexParams(20, 10, 2));
	vector<DMatch> matches;

	//将找到的描述子进行匹配并存入matches中
	fbmatcher.match(descriptor_obj, descriptor_scene, matches);
	double minDist = 1000;
	double maxDist = 0;

	//找出最优描述子
	vector<DMatch> goodmatches;
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < minDist)
		{
			minDist = dist;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
		}

	}
	for (int i = 0; i < descriptor_obj.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < max(2 * minDist, 0.03))
		{
			goodmatches.push_back(matches[i]);
		}
	}

	vector<int>trainIdxs(goodmatches.size());
	vector<int>queryIdxs(goodmatches.size());

	for (size_t i = 0; i < goodmatches.size(); i++)
	{
		//取出训练图片中匹配的点对的索引即id号
		trainIdxs[i] = goodmatches[i].trainIdx;
		queryIdxs[i] = goodmatches[i].queryIdx;
	}

	vector<Point2f> newPoint;
	vector<Point2f> newPoint2;

	//索引值转point2f
	KeyPoint::convert(keypoints_scene, newPoint, trainIdxs);
	KeyPoint::convert(keypoints_obj, newPoint2, queryIdxs);

	int nums = goodmatches.size();
	
	int *disx = new int[nums];
	int *disy = new int[nums];

	for (size_t i = 0; i < goodmatches.size(); i++) {
		//cout << "newPoint" << newPoint2.at(i).x << endl;
		//disx[i] = newPoint.at(i).x - (newPoint2.at(i).x + preImages.cols/2 - 200) ;
		//disy[i] = newPoint.at(i).y - (newPoint2.at(i).y + preImages.rows/2 -100) ;
		disx[i] = newPoint.at(i).x - newPoint2.at(i).x ;
		disy[i] = newPoint.at(i).y - newPoint2.at(i).y  ;
	}
	

	midFilter(disx, nums);
	midFilter(disy, nums);

	/*for (size_t i = 0; i < goodmatches.size(); i++) {
		cout << disx[i] << " ";
	}*/

	cout << " matches.size=" << nums << endl;
	int avgx = -267, avgy = -267;
	int sumx = 0, sumy = 0;
	for (size_t i = 0; i < goodmatches.size(); i++) {
		sumx += disx[i];
		sumy += disy[i];
	}

	if(nums != 0){
	avgx = sumx  / nums ;
	avgy = sumy  / nums;
	}

	cout << "x移动了 " << -avgx << "个像素  y移动了 " << avgy << "个像素" << endl;
	double tx = getTickCount();
	
	//画点
	for (auto iter = newPoint.cbegin(); iter != newPoint.cend(); iter++) {
		circle(images, *iter, 1, Scalar(0, 255, 0), -1);
	}

	
	imshow("ORBdemo", images);
	//Mat orbImg;
	
	/*drawMatches(img1, keypoints_obj, img2, keypoints_scene, goodmatches, orbImg,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		*/
	double tt = getTickCount();
	double t3 = (tt - t1) * 1000 / getTickFrequency();
	double txr = (tt - tx) * 1000 / getTickFrequency();

	cout << "渲染花费时间" << txr << endl;

	cout << "一共花费时间" << t3 << endl;
}

//中值滤波
int *midFilter(int move[], int size) {
	if (size < 3)
		return move;
	int temp[3];
	int *p;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < 3; j++) {
			if ((j + i) < size)
				temp[j] = move[j + i - 1];
			else
				temp[j] = move[i + j - size - 1];
		}
		p = sort(temp, 3);
		move[i] = p[1];
	}
	return move;
}

//冒泡排序
int *sort(int *src, int size) {
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size - i - 1; j++)
		{
			if (src[j] < src[j + 1])
			{
				int temp;
				temp = src[j];
				src[j] = src[j + 1];
				src[j + 1] = temp;
			}
		}
	}
	return src;
}

bool myFAST(Mat image, Point point) {
	//if(image.ptr<uchar>(point.x))
	return 0;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
