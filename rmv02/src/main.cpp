#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

//grey picture
int main()
{
	//读取图片（使用图片的绝对路径）
	cv::Mat src = imread("/home/wwd/rmv02/resources/test_image.png",IMREAD_GRAYSCALE);

	imshow("ImputImage", src);
	
    imwrite("/home/wwd/rmv02/resources/01.png",src);//保存图像

	cv::Mat img = cv::imread("/home/wwd/rmv02/resources/test_image.png");
	cv::Mat HSV;
    cv::cvtColor(img, HSV, cv::COLOR_BGR2HSV);
	imshow("ImputImage", HSV);
    imwrite("/home/wwd/rmv02/resources/02.png",HSV);//保存图像

	cv::Mat junzhi;
	blur(img, junzhi, Size(3, 3));
	imshow("ImputImage", junzhi);
	imwrite("/home/wwd/rmv02/resources/03.png",junzhi);
	cv::Mat gaosi;
	GaussianBlur(img, gaosi, Size(5, 5), 0, 0);
	imshow("ImputImage", gaosi);
	imwrite("/home/wwd/rmv02/resources/04.png",gaosi);

	// 在HSV空间中定义红色,红色的h值有两个范围[0,10]和[156,180]
    cv::Scalar lower_red_1 = cv::Scalar(0, 50, 50);
    cv::Scalar upper_red_1 = cv::Scalar(10, 255, 255);
    cv::Scalar lower_red_2 = cv::Scalar(156, 50, 50);
    cv::Scalar upper_red_2 = cv::Scalar(180, 255, 255);
	// 从HSV图像中截取出红色，即获得相应的掩膜
	cv::Mat red_mask, red_mask_1, red_mask_2;
	cv::inRange(HSV, lower_red_1, upper_red_1, red_mask_1);
    cv::inRange(HSV, lower_red_2, upper_red_2, red_mask_2);
    red_mask = red_mask_1 + red_mask_2;
    cv::Mat red_res;
    cv::bitwise_and(img,img, red_res, red_mask);
    cv::Mat background_img = cv::Mat::zeros(2500, 1900, CV_8UC3);
	//std::cout << "Image size: " << background_img.cols << "x" << background_img.rows << std::endl;


  {
    int x = 100, y = 40;
    cv::Rect roi(x, y, img.cols, img.rows);
    cv::putText(background_img, "origin", cv::Point(x, y - 10),
                cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 1);
    //将background_img复制到img中roi指定的矩形位置
    img.copyTo(background_img(roi));
  }
  {
    int x = 100, y = 40;
    cv::Rect roi(x, y, red_res.cols, red_res.rows);
    cv::putText(background_img, "HSV", cv::Point(x, y - 10),
                cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 1);
    HSV.copyTo(background_img(roi));
  }
  {
    int x = 100, y = 40;
    cv::Rect roi(x, y, red_res.cols, red_res.rows);
    cv::putText(background_img, "red", cv::Point(x, y - 10),
                cv::FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255), 1);
    red_res.copyTo(background_img(roi));
  }
    cv::imwrite("/home/wwd/rmv02/resources/05.png", background_img);
    std::string win_name = "background_img";
    cv::namedWindow(win_name, cv::WINDOW_KEEPRATIO);
    cv::imshow(win_name, background_img);

    //二值化图像 
	cv::Mat erzhi;
	threshold(src, erzhi, 128, 255, THRESH_BINARY);
	//查找轮廓 
	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(erzhi, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (!contours.empty() && !hierarchy.empty()) {
        int i = 1;
        for (size_t idx = 0; idx < contours.size(); idx++) {
            double area = contourArea(contours[idx]);
			// 过滤掉面积过小的轮廓（比如面积小于1的部分）
            if (area > 1){
				cout << "第" << i << "个红色轮廓的面积为：" << area << endl;
                i++;
				}
        }
    }
    //绘制轮廓 
    cv::Mat result(src.size(), CV_8UC3, Scalar(0, 0, 0));
    for (size_t i = 0; i < contours.size(); i++)
    {
        Scalar color(255,0,0);
        drawContours(result, contours, static_cast<int>(i), color, 2, LINE_8, hierarchy, 0);
    }
	 // 绘制矩形框
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) > 1200) {
            cv::Rect boundingBox = cv::boundingRect(contours[i]);
            // 绘制矩形框
            cv::rectangle(result, boundingBox, cv::Scalar(0, 255, 0), 1);
        }
    }

    imshow("轮廓图", result);
	//cv::imwrite("/home/wwd/rmv02/resources/06.png", result);

	// 设置亮度的阈值
    Scalar lowerThresh(0, 0, 200);  // 较高的亮度范围
    Scalar upperThresh(255, 255, 255);
	// 创建掩码，提取高亮区域
    cv::Mat mask1;
    inRange(HSV, lowerThresh, upperThresh, mask1);
	// 使用掩码提取高亮部分
    cv::Mat highLighted;
    bitwise_and(img, img, highLighted, mask1);
	cv::imshow("High Lighted Part", highLighted); // 高亮部分
	cv::imwrite("/home/wwd/rmv02/resources/07.png",highLighted);

	cv::Mat grayHighLighted;
    cvtColor(highLighted, grayHighLighted, COLOR_BGR2GRAY);
    cv::imshow("gray highlighted",grayHighLighted);
	cv::imwrite("/home/wwd/rmv02/resources/08.png", grayHighLighted);

	cv::Mat erzhiHighLighted;
	threshold(grayHighLighted,erzhiHighLighted, 128, 255, THRESH_BINARY);
	cv::imshow("erzhi highlighted",erzhiHighLighted);
	cv::imwrite("/home/wwd/rmv02/resources/09.png", erzhiHighLighted);

     //获取自定义核
	cv::Mat element = getStructuringElement(MORPH_RECT,Size(15,15));
	cv::Mat expansion;
	//进行膨胀操作
	cv::dilate(highLighted,expansion,element);
	cv::imshow("expansion",expansion);
	cv::imwrite("/home/wwd/rmv02/resources/10.png", expansion); 

	cv::Mat ero;
	erode(highLighted,ero,element);
	cv::imshow("erode",ero);
	cv::imwrite("/home/wwd/rmv02/resources/11.png", ero); 

	//对灰度化图像进行漫水处理 
	Rect ccomp;
    floodFill(grayHighLighted, Point(50,300), Scalar(155, 255,55), &ccomp, Scalar(20, 20, 20),Scalar(20, 20, 20));
    cv::imshow("floodfill",grayHighLighted);
	cv::imwrite("/home/wwd/rmv02/resources/12.png", grayHighLighted);

	//绘制图形 
	Point p[1][5] = {{Point(100, 100), Point(150, 50), Point(200, 100),Point(250,200),Point(50, 200)}};
	const Point* pp[] = { p[0]};
	int n[] = {5};
	fillPoly(highLighted, pp, n, 1, Scalar(255, 0, 0));
	//绘制文字 
	putText(highLighted, "hello Natsume", Point(370, 350), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, 8, false);
	imshow("paint", highLighted);
	imwrite("/home/wwd/rmv02/resources/13.png", highLighted);

    // 获取图像的中心点
	cv::Point2f center(highLighted.cols / 2.0, highLighted.rows / 2.0);
	// 计算旋转矩阵，旋转角度为35度，缩放系数为1.0
	double angle = 35.0;
	double scale = 1.0;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, scale);
	// 计算旋转后的图像边界大小
	// 创建旋转后的图像的四个角点，以确保整个图像能被包含在新的边界框内
	std::vector<cv::Point2f> corners(4);
	corners[0] = cv::Point2f(0, 0); // 左上角
	corners[1] = cv::Point2f(highLighted.cols, 0); // 右上角
	corners[2] = cv::Point2f(0, highLighted.rows); // 左下角
	corners[3] = cv::Point2f(highLighted.cols, highLighted.rows); // 右下角 
	// 通过旋转矩阵对角点进行变换，获得旋转后的图像范围
	std::vector<cv::Point2f> rotatedCorners(4);
	cv::transform(corners, rotatedCorners, rotationMatrix);
	// 计算旋转后图像的边界框
	cv::Rect2f bbox = cv::boundingRect(rotatedCorners);
    // 调整旋转矩阵以考虑图像的平移
	rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - highLighted.cols / 2.0;
	rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - highLighted.rows / 2.0;	
	// 应用仿射变换并调整图像大小
	cv::Mat rotated;
	cv::warpAffine(highLighted, rotated, rotationMatrix, bbox.size());
    imshow("rotate", rotated);
	imwrite("/home/wwd/rmv02/resources/14.png", rotated);

	cv::Mat image_part = highLighted(cv::Rect(0,0,855,1213.5)); // 裁剪后的图
	cv::imshow("part of original image", image_part);
	imwrite("/home/wwd/rmv02/resources/15.png", image_part);
   
	waitKey(0);
	getchar();
    return 0;
}